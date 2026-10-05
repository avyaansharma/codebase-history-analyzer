import os
import sys
import re
import json
import subprocess
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional
from sqlmodel import select

from archaeologist.storage.db import get_session_context
from archaeologist.storage.models import Commit, PullRequest, Issue, Chunk, SymbolIndex
from archaeologist.retrieval.embedder import Embedder
from archaeologist.retrieval.vector_store import VectorStore
from archaeologist.retrieval.bm25_index import BM25Index
from archaeologist.retrieval.fusion import reciprocal_rank_fusion
from archaeologist.agent.graph import agent_graph
from archaeologist.utils.gemini_client import GeminiClientWrapper, get_gemini_api_key
from archaeologist.utils.security import validate_repo_path, sanitize_file_path

def _get_json_list(value: Any) -> List[str]:
    """Safely deseralizes JSON list fields stored in SQLite."""
    if not value:
        return []
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return []
    return value

def search_history_tool(
    query: str,
    file_path: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    source_types: Optional[List[str]] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Hybrid search over commit/PR/issue history for a repo."""
    bm25 = BM25Index()
    from archaeologist.storage.paths import get_default_bm25_path
    custom_bm25 = os.getenv("BM25_INDEX_PATH")
    bm25_path = custom_bm25 or ("bm25_index.bin" if os.path.exists("bm25_index.bin") else get_default_bm25_path())
    sparse_hits = []
    if os.path.exists(bm25_path):
        bm25.load(bm25_path)
        sparse_hits = bm25.search(query, limit=30, file_path=file_path, source_types=source_types)

    embedder = Embedder()
    vector_store = VectorStore(vector_size=embedder.dimension)
    dense_hits = []
    try:
        query_vectors = embedder.embed_texts([query])
        if query_vectors:
            dense_hits = vector_store.search_chunks(
                query_vector=query_vectors[0],
                limit=30,
                file_path=file_path,
                date_from=date_from,
                date_to=date_to,
                source_types=source_types
            )
    except Exception as e:
        print(f"Error in vector search: {e}", file=sys.stderr)
    finally:
        vector_store.close()

    fused_results = reciprocal_rank_fusion(dense_hits, sparse_hits, limit=limit)
    
    formatted = []
    for hit in fused_results:
        formatted.append({
            "id": hit["id"],
            "rrf_score": hit["rrf_score"],
            "payload": hit["payload"]
        })
    return formatted

def find_related_discussion_tool(ref: str) -> Dict[str, Any]:
    """Given a commit SHA or PR/issue number, return linked issues, PRs, and commits."""
    result = {"commits": [], "pull_requests": [], "issues": []}
    
    with get_session_context() as session:
        ref_cleaned = ref.strip().lower()
        
        from archaeologist.utils.security import SHA_REGEX
        if len(ref_cleaned) >= 7 and not ref_cleaned.startswith("#") and SHA_REGEX.match(ref_cleaned):
            stmt = select(Commit).where(Commit.sha.like(f"{ref_cleaned}%"))
            commit = session.exec(stmt).first()
            if commit:
                result["commits"].append({
                    "sha": commit.sha,
                    "author": commit.author_name,
                    "date": commit.authored_date.isoformat(),
                    "message": commit.message,
                    "is_revert": commit.is_revert,
                    "reverts_sha": commit.reverts_sha,
                    "superseded_by_sha": commit.superseded_by_sha
                })
                
                stmt_chunk = select(Chunk).where(Chunk.source_type == "commit", Chunk.source_id == commit.sha)
                chunks = session.exec(stmt_chunk).all()
                related_refs = set()
                for c in chunks:
                    related_refs.update(c.related_ids)
                
                for r in related_refs:
                    if r.startswith("pr#"):
                        pr_num = int(r.split("#")[-1])
                        pr = session.get(PullRequest, pr_num)
                        if pr:
                            result["pull_requests"].append({"number": pr.number, "title": pr.title, "state": pr.state})
                    elif r.startswith("issue#"):
                        issue_num = int(r.split("#")[-1])
                        issue = session.get(Issue, issue_num)
                        if issue:
                            result["issues"].append({"number": issue.number, "title": issue.title, "state": issue.state})

        else:
            num_match = re.search(r'\d+', ref_cleaned)
            if num_match:
                num = int(num_match.group(0))
                
                pr = session.get(PullRequest, num)
                if pr:
                    result["pull_requests"].append({
                        "number": pr.number,
                        "title": pr.title,
                        "state": pr.state,
                        "author": pr.author,
                        "linked_issues": pr.linked_issue_numbers
                    })
                    for i_num in pr.linked_issue_numbers:
                        i = session.get(Issue, i_num)
                        if i:
                            result["issues"].append({"number": i.number, "title": i.title, "state": i.state})
                            
                    # Priority 1: Use pr.linked_commit_shas extracted during ingestion
                    linked_shas = set(pr.linked_commit_shas or [])
                    if linked_shas:
                        stmt_commits = select(Commit).where(Commit.sha.in_(linked_shas))
                        linked_commits = session.exec(stmt_commits).all()
                    else:
                        from archaeologist.utils.security import escape_like
                        escaped_num = escape_like(str(pr.number))
                        stmt_commits = select(Commit).where(Commit.message.like(f"%#{escaped_num}%", escape="\\")).limit(50)
                        candidates = session.exec(stmt_commits).all()
                        linked_commits = [c for c in candidates if re.search(r'#' + re.escape(str(pr.number)) + r'\b', c.message)]
                    
                    for c in linked_commits:
                        result["commits"].append({"sha": c.sha, "message": c.message, "author": c.author_name})
                
                issue = session.get(Issue, num)
                if issue:
                    result["issues"].append({
                        "number": issue.number,
                        "title": issue.title,
                        "state": issue.state,
                        "labels": issue.labels,
                        "linked_prs": issue.linked_pr_numbers
                    })
                    for p_num in issue.linked_pr_numbers:
                        p = session.get(PullRequest, p_num)
                        if p:
                            result["pull_requests"].append({"number": p.number, "title": p.title, "state": p.state})

                    # Priority 1: Use issue.linked_commit_shas extracted during ingestion
                    linked_shas = set(issue.linked_commit_shas or [])
                    if linked_shas:
                        stmt_commits = select(Commit).where(Commit.sha.in_(linked_shas))
                        linked_commits = session.exec(stmt_commits).all()
                    else:
                        from archaeologist.utils.security import escape_like
                        escaped_num = escape_like(str(issue.number))
                        stmt_commits = select(Commit).where(Commit.message.like(f"%#{escaped_num}%", escape="\\")).limit(50)
                        candidates = session.exec(stmt_commits).all()
                        linked_commits = [c for c in candidates if re.search(r'#' + re.escape(str(issue.number)) + r'\b', c.message)]


                    for c in linked_commits:
                        if not any(existing["sha"] == c.sha for existing in result["commits"]):
                            result["commits"].append({"sha": c.sha, "message": c.message, "author": c.author_name})

    return result

def blame_explain_tool(
    repo_path: str,
    file_path: str,
    line_start: int,
    line_end: int
) -> Dict[str, Any]:
    """Given a file and line range, return the commit history and causal explanation for why that code exists."""
    validated_repo = validate_repo_path(repo_path)
    clean_file_path = sanitize_file_path(validated_repo, file_path)

    cmd = [
        "git", "-C", validated_repo, "blame", 
        "-L", f"{line_start},{line_end}", 
        "--porcelain", clean_file_path
    ]
    
    shas = set()
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if res.returncode == 0:
            for line in res.stdout.splitlines():
                parts = line.split()
                if parts and len(parts[0]) == 40:
                    shas.add(parts[0])
    except Exception as e:
        print(f"Error running git blame: {e}", file=sys.stderr)

    commits_data = []
    with get_session_context() as session:
        for sha in shas:
            c = session.get(Commit, sha)
            if c:
                commits_data.append({
                    "sha": c.sha,
                    "author": c.author_name,
                    "date": c.authored_date.isoformat(),
                    "message": c.message,
                    "is_revert": c.is_revert
                })

    gemini = GeminiClientWrapper()
    prompt = f"Analyze the following commits and explain why lines {line_start}-{line_end} in {clean_file_path} were created or modified:\n{json.dumps(commits_data, indent=2)}"
    explanation = gemini.generate_text(prompt)
    
    return {
        "file_path": clean_file_path,
        "line_range": f"{line_start}-{line_end}",
        "commits": commits_data,
        "explanation": explanation
    }

def repo_hotspots_tool(top_n: int = 15, repo_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Calculates top churn files in the repo ranked by commit count."""
    file_counts = Counter()
    with get_session_context() as session:
        query = select(Commit)
        if repo_id:
            query = query.where(Commit.repo_id == repo_id)
        commits = session.exec(query).all()
        for c in commits:
            files = _get_json_list(c.files_changed)
            for f in files:
                file_counts[f] += 1
    
    hotspots = []
    for fpath, count in file_counts.most_common(top_n):
        hotspots.append({"file_path": fpath, "commit_count": count})
    return hotspots

def repo_ownership_tool(file_path: Optional[str] = None, repo_id: Optional[str] = None) -> Dict[str, Any]:
    """Calculates author percentage contribution distribution and bus factor risk."""
    author_counts = Counter()
    file_author_counts = defaultdict(Counter)
    total_commits = 0
    with get_session_context() as session:
        query = select(Commit)
        if repo_id:
            query = query.where(Commit.repo_id == repo_id)
        commits = session.exec(query).all()
        for c in commits:
            files = _get_json_list(c.files_changed)
            if not file_path or file_path in files:
                author_counts[c.author_name] += 1
                total_commits += 1
            for f in files:
                file_author_counts[f][c.author_name] += 1

    distribution = {}
    for author, count in author_counts.items():
        distribution[author] = {
            "commit_count": count,
            "percentage": round((count / total_commits) * 100, 2) if total_commits > 0 else 0.0
        }

    # Calculate bus factor risk (e.g. HIGH if single author owns > 60% of commits)
    max_pct = max([data["percentage"] for data in distribution.values()]) if distribution else 0.0
    risk = "HIGH (Single Author Dominance)" if max_pct > 60.0 else "NORMAL"

    res = {
        "target_file": file_path or "GLOBAL REPOSITORY",
        "total_commits": total_commits,
        "author_distribution": distribution,
        "bus_factor_risk": risk
    }

    if not file_path:
        file_breakdown = {}
        sorted_files = sorted(file_author_counts.items(), key=lambda item: sum(item[1].values()), reverse=True)
        for f, counts in sorted_files[:20]:
            f_total = sum(counts.values())
            top_author, top_cnt = counts.most_common(1)[0]
            f_pct = round((top_cnt / f_total) * 100, 2) if f_total > 0 else 0
            file_breakdown[f] = {
                "total_commits": f_total,
                "top_author": top_author,
                "top_author_pct": f_pct,
                "bus_factor_risk": "HIGH" if f_pct > 60.0 else "NORMAL"
            }
        res["per_file_breakdown"] = file_breakdown

    return res


def change_coupling_tool(min_co_commits: int = 2, top_n: int = 15, repo_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Identifies pairs of files that frequently change together in the same commit."""
    pair_counts = Counter()
    with get_session_context() as session:
        query = select(Commit)
        if repo_id:
            query = query.where(Commit.repo_id == repo_id)
        commits = session.exec(query).all()
        for c in commits:
            files = sorted(list(set(_get_json_list(c.files_changed))))
            if len(files) > 15:
                files = files[:15]
            for i in range(len(files)):
                for j in range(i + 1, len(files)):
                    pair_counts[(files[i], files[j])] += 1

    couplings = []
    for (f1, f2), count in pair_counts.most_common(top_n):
        if count >= min_co_commits:
            couplings.append({
                "file_a": f1,
                "file_b": f2,
                "co_commit_count": count
            })
    return couplings

def repo_symbols_tool(top_n: int = 20, repo_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Lists extracted AST code symbols (classes, functions, methods) ranked by modification frequency."""
    with get_session_context() as session:
        query = select(SymbolIndex).order_by(SymbolIndex.commit_count.desc())
        if repo_id:
            query = query.where(SymbolIndex.repo_id == repo_id)
        symbols = session.exec(query.limit(top_n)).all()
        return [
            {
                "symbol_id": s.symbol_id,
                "file_path": s.file_path,
                "symbol_name": s.symbol_name,
                "kind": s.kind,
                "commit_count": s.commit_count
            }
            for s in symbols
        ]

def symbol_history_tool(symbol_query: str, repo_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves all commits that modified a specific AST Code Symbol (e.g. 'AuthService' or 'login')."""
    matching_commits = []
    with get_session_context() as session:
        query = select(Commit)
        if repo_id:
            query = query.where(Commit.repo_id == repo_id)
        commits = session.exec(query).all()
        for c in commits:
            syms = _get_json_list(c.symbols_modified)
            if any(symbol_query.lower() in s.lower() for s in syms):
                matching_commits.append({
                    "sha": c.sha,
                    "author": c.author_name,
                    "date": c.authored_date.isoformat(),
                    "message": c.message,
                    "symbols_modified": syms,
                    "files_changed": _get_json_list(c.files_changed)
                })
    return matching_commits

def ask_tool(question: str, repo_id: Optional[str] = None) -> str:
    """Answers a causal 'why' question about the codebase using full agentic multi-hop retrieval."""
    inputs = {
        "question": question,
        "repo_id": repo_id,
        "sub_questions": [],
        "current_sub_question_index": 0,
        "search_queries": [],
        "retrieved_chunks": [],
        "evidence_by_chunk_id": {},
        "draft_answer": None,
        "verification_passed": False,
        "unverified_claims": [],
        "retry_count": 0,
        "response": None
    }
    
    try:
        final_state = agent_graph.invoke(inputs, config={"recursion_limit": 60})
        return final_state.get("response", "Could not synthesize response.")
    except Exception as e:
        return f"Error executing agent loop: {e}"
