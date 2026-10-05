import os
import sys
import re
from datetime import datetime
from typing import Optional, List, Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlmodel import select

from archaeologist.storage.db import init_db, get_session_context
from archaeologist.storage.paths import get_default_bm25_path
from archaeologist.storage.models import Commit, PullRequest, Issue, Chunk, SymbolIndex
from archaeologist.ingestion.git_parser import iter_commits, count_commits, get_commit_diff, is_merge_commit
from archaeologist.ingestion.revert_detector import detect_revert_from_message, find_reverted_commit
from archaeologist.ingestion.github_client import GitHubIngestionClient
from archaeologist.ingestion.link_resolver import update_cross_links
from archaeologist.ingestion.chunker import chunk_commit, chunk_issue, chunk_pr, chunk_codebase, make_deterministic_chunk_id, token_count

from archaeologist.ingestion.diff_summarizer import LLMSummarizer
from archaeologist.ingestion.symbol_parser import (
    extract_symbols_from_code,
    map_lines_to_symbols,
    extract_modified_line_numbers_from_diff,
    resolve_file_path,
    get_git_file_content
)
from archaeologist.retrieval.embedder import Embedder
from archaeologist.retrieval.vector_store import VectorStore
from archaeologist.retrieval.bm25_index import BM25Index

class IngestionPipeline:
    def __init__(
        self,
        repo_path: str,
        repo_url: Optional[str] = None,
        since_date: Optional[str] = None,
        github_limit: int = 500,
        repo_id: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int, int, str, float], None]] = None,
        github_token: Optional[str] = None,
        gemini_key: Optional[str] = None
    ):
        self.repo_path = repo_path
        self.repo_url = repo_url
        self.since_date = since_date
        self.github_limit = github_limit
        self.progress_callback = progress_callback
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")
        self.gemini_key = gemini_key or os.getenv("GEMINI_API_KEY")
        if repo_id:
            self.repo_id = repo_id
        elif repo_url:
            parts = repo_url.rstrip("/").split("/")
            self.repo_id = parts[-1].lower() if len(parts) >= 1 else os.path.basename(os.path.abspath(repo_path)).lower()
        else:
            self.repo_id = os.path.basename(os.path.abspath(repo_path)).lower()

    def _report_progress(self, phase: str, current: int, total: int, message: str, percent: float):
        """Dispatches real-time percentage progress updates to active progress_callback or stderr."""
        bounded_pct = min(100.0, max(0.0, float(percent)))
        if self.progress_callback:
            try:
                self.progress_callback(phase, current, total, message, bounded_pct)
            except Exception:
                pass
        else:
            print(f"[{bounded_pct:3.0f}%] {message}", file=sys.stderr)

    def run(self):
        from archaeologist.storage.paths import get_default_db_url, get_default_bm25_path
        repo_db_url = get_default_db_url(self.repo_path)
        os.environ["DATABASE_URL"] = repo_db_url
        os.environ["BM25_INDEX_PATH"] = get_default_bm25_path(self.repo_path)
        
        self._report_progress("init", 0, 100, "Initializing metadata database...", 2.0)
        init_db(repo_db_url)

        self._report_progress("init", 50, 100, "Initializing vector store collection...", 5.0)
        embedder = Embedder()
        vector_store = VectorStore(vector_size=embedder.dimension)
        vector_store.init_collection()
        vector_store.close()

        # Step 1: Walk git log & Extract AST Symbol Graph at historical commit SHAs
        total_commits = count_commits(self.repo_path, since=self.since_date)
        if total_commits == 0:
            total_commits = 1
            
        self._report_progress("commits", 0, total_commits, f"Scanning {total_commits} commits...", 6.0)
        new_commits_count = 0
        
        with get_session_context() as session:
            query = select(Commit.sha)
            if self.repo_id:
                query = query.where(Commit.repo_id == self.repo_id)
            existing_shas = set(session.exec(query).all())
            
            c_idx = 0
            for c_data in iter_commits(self.repo_path, since=self.since_date):
                c_idx += 1
                if c_data["sha"] in existing_shas:
                    continue
                    
                reverted_subject = detect_revert_from_message(c_data["message"])
                is_revert = bool(reverted_subject)
                
                # AST Symbol Extraction Pass at Historical Commit SHA
                symbols_modified = []
                diff_text = get_commit_diff(self.repo_path, c_data["sha"])
                if diff_text:
                    modified_lines_map = extract_modified_line_numbers_from_diff(diff_text)
                    for fpath, line_buckets in modified_lines_map.items():
                        added_lines = line_buckets.get("added", [])
                        deleted_lines = line_buckets.get("deleted", [])
                        touched_syms = []

                        # Map added lines against post-commit file content
                        post_code_text = get_git_file_content(self.repo_path, c_data["sha"], fpath)
                        if not post_code_text:
                            resolved_path = resolve_file_path(self.repo_path, fpath)
                            if resolved_path:
                                try:
                                    with open(resolved_path, "r", encoding="utf-8", errors="ignore") as f:
                                        post_code_text = f.read()
                                except Exception:
                                    post_code_text = None

                        if post_code_text:
                            post_symbols = extract_symbols_from_code(post_code_text, fpath)
                            if added_lines:
                                touched_syms.extend(map_lines_to_symbols(post_symbols, added_lines))

                            for sym in post_symbols:
                                sym_obj = session.get(SymbolIndex, sym["symbol_id"])
                                if not sym_obj:
                                    sym_obj = SymbolIndex(
                                        symbol_id=sym["symbol_id"],
                                        repo_id=self.repo_id,
                                        file_path=fpath,
                                        symbol_name=sym["name"],
                                        kind=sym["kind"],
                                        commit_count=0
                                    )
                                    session.add(sym_obj)

                        # Map deleted lines against pre-commit file content
                        if deleted_lines:
                            pre_code_text = get_git_file_content(self.repo_path, f"{c_data['sha']}^", fpath)
                            if pre_code_text:
                                pre_symbols = extract_symbols_from_code(pre_code_text, fpath)
                                touched_syms.extend(map_lines_to_symbols(pre_symbols, deleted_lines))

                        unique_touched = sorted(list(set(touched_syms)))
                        symbols_modified.extend(unique_touched)

                        for sym_id in unique_touched:
                            sym_obj = session.get(SymbolIndex, sym_id)
                            if sym_obj:
                                sym_obj.commit_count += 1
                                session.add(sym_obj)


                commit_obj = Commit(
                    sha=c_data["sha"],
                    repo_id=self.repo_id,
                    author_name=c_data["author_name"],
                    author_email=c_data["author_email"],
                    authored_date=c_data["authored_date"],
                    message=c_data["message"],
                    files_changed=c_data["files_changed"],
                    symbols_modified=sorted(list(set(symbols_modified))),
                    insertions=c_data["insertions"],
                    deletions=c_data["deletions"],
                    is_revert=is_revert,
                    reverts_sha=None
                )
                session.add(commit_obj)
                existing_shas.add(c_data["sha"])
                new_commits_count += 1
                
                c_pct = 6.0 + min(1.0, c_idx / total_commits) * 28.0
                if c_idx % max(1, total_commits // 40) == 0 or c_idx == total_commits:
                    self._report_progress(
                        "commits",
                        c_idx,
                        total_commits,
                        f"Parsed commit {c_idx}/{total_commits} ({c_data['sha'][:7]})",
                        c_pct
                    )
                
        self._report_progress("commits", total_commits, total_commits, f"Ingested {new_commits_count} commits into database", 35.0)

        # Step 1b: Codebase File Ingestion Pass
        self._report_progress("files", 0, 1, "Scanning codebase source files...", 36.0)
        all_chunks = []

        with get_session_context() as session:
            for root, dirs, files in os.walk(self.repo_path):
                dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("venv", ".venv", "__pycache__", "node_modules", "site-packages", "dist", "build")]
                for fname in files:
                    if fname.endswith((".py", ".md", ".json", ".ts", ".js", ".yml", ".yaml", ".jsonl")):
                        abs_fpath = os.path.join(root, fname)
                        rel_fpath = os.path.relpath(abs_fpath, self.repo_path).replace("\\", "/")
                        try:
                            with open(abs_fpath, "r", encoding="utf-8", errors="ignore") as f:
                                content = f.read()
                            if content.strip():
                                chunk_id = make_deterministic_chunk_id("file", rel_fpath, 0, content)
                                c_obj = session.get(Chunk, chunk_id)
                                if not c_obj:
                                    chunk_text = f"File {rel_fpath}:\n{content[:3000]}"
                                    all_chunks.append({
                                        "id": chunk_id,
                                        "source_type": "file",
                                        "source_id": rel_fpath,
                                        "text": chunk_text,
                                        "timestamp": datetime.utcnow(),
                                        "file_paths": [rel_fpath],
                                        "symbols_modified": [],
                                        "is_reverted": False,
                                        "token_count": token_count(chunk_text),
                                        "embedded": False
                                    })
                        except Exception:
                            pass
            session.add_all([Chunk(**c) for c in all_chunks])
        self._report_progress("files", len(all_chunks), len(all_chunks) or 1, f"Extracted {len(all_chunks)} source code files", 38.0)

        # Step 2: Revert Detection Resolution Pass
        self._report_progress("reverts", 0, 1, "Detecting revert chains and supersessions...", 39.0)
        with get_session_context() as session:
            rev_query = select(Commit).where(Commit.is_revert == True)
            if self.repo_id:
                rev_query = rev_query.where(Commit.repo_id == self.repo_id)
            revert_commits = session.exec(rev_query).all()
            reverts_resolved = 0
            for r_commit in revert_commits:
                if not r_commit.reverts_sha:
                    reverted_subject = detect_revert_from_message(r_commit.message)
                    if reverted_subject:
                        orig_commit = find_reverted_commit(session, reverted_subject, r_commit.authored_date)
                        if orig_commit:
                            r_commit.reverts_sha = orig_commit.sha
                            session.add(r_commit)
                            orig_commit.superseded_by_sha = r_commit.sha
                            session.add(orig_commit)
                            reverts_resolved += 1
        self._report_progress("reverts", 1, 1, f"Resolved {reverts_resolved} revert chains", 41.0)

        # Step 3: Fetch GitHub Issues and PRs (Historical order: direction='asc')
        if self.repo_url and self.github_limit > 0 and not os.getenv("SKIP_GITHUB_API"):
            try:
                self._report_progress("github", 0, 1, f"Checking GitHub PRs & Issues for {self.repo_url}...", 42.0)

                gh_client = GitHubIngestionClient(self.repo_url, token=self.github_token)
                prs = gh_client.fetch_pull_requests(limit=self.github_limit, direction="asc")
                issues = gh_client.fetch_issues(limit=self.github_limit, direction="asc")
                
                with get_session_context() as session:
                    for pr_data in prs:
                        pr_obj = session.get(PullRequest, pr_data["number"])
                        if not pr_obj:
                            pr_obj = PullRequest(
                                number=pr_data["number"],
                                repo_id=self.repo_id,
                                title=pr_data["title"],
                                body=pr_data["body"],
                                state=pr_data["state"],
                                author=pr_data.get("author", "unknown"),
                                created_at=pr_data["created_at"],
                                merged_at=pr_data["merged_at"],
                                merge_commit_sha=pr_data.get("merged_commit_sha"),
                                comments=pr_data.get("comments", []),
                                review_comments=pr_data.get("review_comments", []),
                                linked_issue_numbers=pr_data.get("linked_issue_numbers", [])
                            )
                            session.add(pr_obj)

                    for issue_data in issues:
                        issue_obj = session.get(Issue, issue_data["number"])
                        if not issue_obj:
                            issue_obj = Issue(
                                number=issue_data["number"],
                                repo_id=self.repo_id,
                                title=issue_data["title"],
                                body=issue_data["body"],
                                state=issue_data["state"],
                                author=issue_data.get("author", "unknown"),
                                created_at=issue_data["created_at"],
                                closed_at=issue_data.get("closed_at"),
                                labels=issue_data.get("labels", []),
                                comments=issue_data.get("comments", []),
                                linked_pr_numbers=issue_data.get("linked_pr_numbers", [])
                            )
                            session.add(issue_obj)
                self._report_progress("github", 1, 1, "GitHub metadata synchronized", 44.0)
            except Exception as e:
                self._report_progress("github", 1, 1, f"Skipped GitHub API ({e})", 44.0)
        else:
            self._report_progress("github", 1, 1, "Skipping remote GitHub API ingestion", 44.0)

        # Step 4: Cross-Link Resolution Pass
        self._report_progress("links", 0, 1, "Updating commit cross-links...", 45.0)
        with get_session_context() as session:
            update_cross_links(session)
        self._report_progress("links", 1, 1, "Commit cross-links synchronized", 47.0)

        # Step 5: Batched Chunking & Summarization Pass
        summarizer = LLMSummarizer()
        
        with get_session_context() as session:
            c_query = select(Commit)
            pr_query = select(PullRequest)
            if self.repo_id:
                c_query = c_query.where(Commit.repo_id == self.repo_id)
                pr_query = pr_query.where(PullRequest.repo_id == self.repo_id)
            all_commits = session.exec(c_query).all()
            all_prs = session.exec(pr_query).all()
            
            pr_by_commit = {}
            for p in all_prs:
                if p.merged_commit_sha:
                    pr_by_commit[p.merged_commit_sha] = p.number
                for sha in p.linked_commit_shas:
                    pr_by_commit[sha] = p.number

            eligible_diffs = []
            for c in list(reversed(all_commits))[:100]:
                if 0 < len(c.files_changed) <= 20:
                    d_text = get_commit_diff(self.repo_path, c.sha)
                    if d_text:
                        eligible_diffs.append({"sha": c.sha, "diff_text": d_text})

            diff_summaries = {}
            batch_size = 5
            total_diff_batches = max(1, (len(eligible_diffs) + batch_size - 1) // batch_size) if eligible_diffs else 1
            self._report_progress("diffs", 0, total_diff_batches, f"Summarizing {len(eligible_diffs)} diffs in {total_diff_batches} batches...", 48.0)

            for b_idx, i in enumerate(range(0, len(eligible_diffs), batch_size), 1):
                batch = eligible_diffs[i:i + batch_size]
                batch_res = summarizer.summarize_diff_batch(batch)
                diff_summaries.update(batch_res)
                diff_pct = 48.0 + (b_idx / total_diff_batches) * 23.0
                self._report_progress("diffs", b_idx, total_diff_batches, f"Summarized diff batch {b_idx}/{total_diff_batches}", diff_pct)

            for c in all_commits:
                summary = diff_summaries.get(c.sha)
                commit_dict = c.model_dump()
                related = []
                if c.sha in pr_by_commit:
                    related.append(f"pr#{pr_by_commit[c.sha]}")
                commit_dict["related_ids"] = related

                commit_chunks = chunk_commit(commit=commit_dict, diff_summary=summary, repo_id=self.repo_id)
                if isinstance(commit_chunks, dict):
                    commit_chunks = [commit_chunks]
                for chunk_info in commit_chunks:
                    c_obj = session.get(Chunk, chunk_info["id"])
                    if not c_obj:
                        session.add(Chunk(**chunk_info))

            for pr in all_prs:
                pr_dict = pr.model_dump()
                pr_chunks = chunk_pr(pr=pr_dict, repo_id=self.repo_id)
                for chunk_info in pr_chunks:
                    c_obj = session.get(Chunk, chunk_info["id"])
                    if not c_obj:
                        session.add(Chunk(**chunk_info))

            iss_query = select(Issue)
            if self.repo_id:
                iss_query = iss_query.where(Issue.repo_id == self.repo_id)
            all_issues = session.exec(iss_query).all()
            for i in all_issues:
                issue_dict = i.model_dump()
                issue_chunks = chunk_issue(issue=issue_dict, repo_id=self.repo_id)
                for chunk_info in issue_chunks:
                    c_obj = session.get(Chunk, chunk_info["id"])
                    if not c_obj:
                        session.add(Chunk(**chunk_info))

            code_chunks = chunk_codebase(self.repo_path, repo_id=self.repo_id)
            for chunk_info in code_chunks:
                c_obj = session.get(Chunk, chunk_info["id"])
                if not c_obj:
                    session.add(Chunk(**chunk_info))

        self._report_progress("chunks", 1, 1, "Completed codebase chunking pass", 72.0)

        # Step 6: Indexing (BM25 + Qdrant)
        with get_session_context() as session:
            chk_query = select(Chunk)
            if self.repo_id:
                chk_query = chk_query.where(Chunk.repo_id == self.repo_id)
            all_chunks_db = session.exec(chk_query).all()
            all_chunk_dicts = [
                {
                    "id": c.id,
                    "repo_id": c.repo_id,
                    "source_type": c.source_type,
                    "source_id": c.source_id,
                    "text": c.text,
                    "timestamp": c.timestamp,
                    "file_paths": c.file_paths,
                    "symbols_modified": c.symbols_modified,
                    "related_ids": c.related_ids,
                    "is_reverted": c.is_reverted
                }
                for c in all_chunks_db
            ]

        # 6a. BM25 Sparse Index
        self._report_progress("bm25", 0, len(all_chunk_dicts) or 1, f"Fitting BM25 index on {len(all_chunk_dicts)} chunks...", 74.0)
        if all_chunk_dicts:
            bm25 = BM25Index()
            bm25.fit(all_chunk_dicts)
            bm25_save_path = get_default_bm25_path(self.repo_path)
            bm25.save(bm25_save_path)
            self._report_progress("bm25", len(all_chunk_dicts), len(all_chunk_dicts), "BM25 index saved", 80.0)

        # 6b. Qdrant Dense Index
        with get_session_context() as session:
            unemb_query = select(Chunk).where(Chunk.embedded == False)
            if self.repo_id:
                unemb_query = unemb_query.where(Chunk.repo_id == self.repo_id)
            unembedded_chunks = session.exec(unemb_query).all()
            unembedded_dicts = [
                {
                    "id": c.id,
                    "source_type": c.source_type,
                    "source_id": c.source_id,
                    "text": c.text,
                    "timestamp": c.timestamp,
                    "file_paths": c.file_paths,
                    "symbols_modified": c.symbols_modified,
                    "related_ids": c.related_ids,
                    "is_reverted": c.is_reverted
                }
                for c in unembedded_chunks
            ]

        total_unembedded = len(unembedded_dicts)
        if total_unembedded > 0:
            self._report_progress("embeddings", 0, total_unembedded, f"Generating embeddings for {total_unembedded} chunks...", 83.0)
            texts = [c["text"] for c in unembedded_dicts]
            embeddings, success_flags = embedder.embed_texts(texts, return_success_flags=True)
            
            if embeddings:
                successful_dicts = [d for d, success in zip(unembedded_dicts, success_flags) if success]
                successful_embeddings = [e for e, success in zip(embeddings, success_flags) if success]
                
                if successful_dicts:
                    actual_dim = len(successful_embeddings[0])
                    vector_store = VectorStore(vector_size=actual_dim)
                    vector_store.init_collection()
                    vector_store.upsert_chunks(successful_dicts, successful_embeddings)
                    with get_session_context() as session:
                        for c_dict in successful_dicts:
                            c_db = session.get(Chunk, c_dict["id"])
                            if c_db:
                                c_db.embedded = True
                                session.add(c_db)
                    self._report_progress("embeddings", len(successful_dicts), total_unembedded, f"Vector indexing complete ({len(successful_dicts)}/{total_unembedded})", 98.0)
                    vector_store.close()
                else:
                    self._report_progress("embeddings", 0, total_unembedded, "Quota limit reached; using sparse index fallback", 98.0)
        else:
            self._report_progress("embeddings", 1, 1, "All chunks already indexed in vector store", 98.0)

        self._report_progress("complete", 1, 1, "Ingestion completed successfully!", 100.0)

