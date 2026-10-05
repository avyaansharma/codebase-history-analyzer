import os
import sys
import json
import asyncio
import re
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, Query, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from sqlmodel import select

from archaeologist.web.registry import get_repo_config, list_repo_configs, REPOSITORIES
from archaeologist.storage.db import get_session_context
from archaeologist.storage.models import Commit, PullRequest, Issue, SymbolIndex, Chunk
from archaeologist.mcp_server.tools import (
    repo_hotspots_tool,
    repo_ownership_tool,
    change_coupling_tool,
    repo_symbols_tool,
    symbol_history_tool,
    ask_tool
)
from archaeologist.agent.graph import agent_graph

app = FastAPI(
    title="Codebase History Analyzer Intelligence API",
    description="Autonomous Forensic Code Intelligence & Temporal Causal Graph Exploration API",
    version="3.0.0"
)

# Enable CORS for local dev and hosted web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite"
]

def _activate_repo_environment(repo_id: str, client_api_key: Optional[str] = None):
    """Activates the SQLite database and BM25 index path for the specified repository."""
    config = get_repo_config(repo_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Repository '{repo_id}' not found in registry.")
    
    os.environ["DATABASE_URL"] = f"sqlite:///{config['db_path']}"
    os.environ["BM25_INDEX_PATH"] = config["bm25_path"]
    
    if client_api_key and client_api_key.strip():
        os.environ["GEMINI_API_KEY"] = client_api_key.strip()
    return config

@app.get("/api/repos")
async def get_repositories():
    """Returns all available indexed repositories with statistics and starter questions."""
    return {"repositories": list_repo_configs()}

@app.get("/api/repos/{repo_id}")
async def get_repository_details(repo_id: str):
    """Returns detailed configuration for a specific repository."""
    config = get_repo_config(repo_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Repository '{repo_id}' not found.")
    return config

@app.post("/api/validate-key")
async def validate_gemini_key(payload: Dict[str, Any]):
    """Tests if a user-supplied Gemini API key is valid and active using the configured model tier."""
    api_key = payload.get("api_key", "").strip()
    model = payload.get("model", "gemini-2.5-flash-lite")
    if not api_key:
        raise HTTPException(status_code=400, detail="No API key provided.")
    
    # Ensure selected model is in supported list
    if model not in SUPPORTED_MODELS:
        model = "gemini-2.5-flash-lite"
        
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        # Lightweight test prompt using client-selected model
        response = client.models.generate_content(
            model=model,
            contents="ping"
        )
        return {
            "valid": True,
            "status": "active",
            "model": model,
            "message": f"Gemini API key validated successfully using {model}."
        }
    except Exception as e:
        error_msg = str(e)
        if "400" in error_msg or "API_KEY_INVALID" in error_msg:
            return {"valid": False, "status": "invalid", "message": "Invalid Gemini API Key."}
        elif "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            return {"valid": True, "status": "quota_exhausted", "message": f"Valid key, but currently rate-limited (429) on {model}."}
        else:
            return {"valid": False, "status": "error", "message": f"Validation error: {error_msg[:120]}"}

@app.get("/api/graph/{repo_id}")
async def get_causal_knowledge_graph(
    repo_id: str, 
    limit: int = Query(250, ge=20, le=1000),
    include_all: bool = Query(True)
):
    """Returns comprehensive structured nodes and causal edges across Issues, PRs, Commits, Reverts, and AST Symbols."""
    config = _activate_repo_environment(repo_id)
    
    nodes = []
    edges = []
    node_ids = set()
    edge_set = set()
    
    def add_edge(src, tgt, edge_type, label, highlight=False):
        edge_key = (src, tgt, edge_type)
        if edge_key not in edge_set and src in node_ids and tgt in node_ids:
            edge_set.add(edge_key)
            edges.append({
                "source": src,
                "target": tgt,
                "type": edge_type,
                "label": label,
                "highlight": highlight
            })
    
    with get_session_context() as session:
        # 1. Fetch ALL Issues
        issues = session.exec(select(Issue).order_by(Issue.created_at.desc())).all()
        for iss in issues:
            iss_id = f"issue:#{iss.number}"
            if iss_id not in node_ids:
                node_ids.add(iss_id)
                nodes.append({
                    "id": iss_id,
                    "type": "issue",
                    "label": f"Issue #{iss.number}",
                    "title": iss.title,
                    "author": iss.author or "contributor",
                    "state": iss.state or "closed",
                    "labels": iss.labels or [],
                    "date": iss.created_at.strftime("%Y-%m-%d") if iss.created_at else "",
                    "linked_prs": iss.linked_pr_numbers or [],
                    "linked_commits": iss.linked_commit_shas or []
                })

        # 2. Fetch ALL Pull Requests
        prs = session.exec(select(PullRequest).order_by(PullRequest.created_at.desc())).all()
        for pr in prs:
            pr_id = f"pr:#{pr.number}"
            if pr_id not in node_ids:
                node_ids.add(pr_id)
                nodes.append({
                    "id": pr_id,
                    "type": "pr",
                    "label": f"PR #{pr.number}",
                    "title": pr.title,
                    "author": pr.author or "contributor",
                    "state": pr.state or "merged",
                    "date": pr.created_at.strftime("%Y-%m-%d") if pr.created_at else "",
                    "linked_issues": pr.linked_issue_numbers or [],
                    "linked_commits": pr.linked_commit_shas or []
                })

        # 3. Fetch Reverts & Significant Commits
        revert_stmt = select(Commit).where(
            (Commit.is_revert == True) | (Commit.reverts_sha != None) | (Commit.superseded_by_sha != None)
        )
        revert_commits = session.exec(revert_stmt).all()
        
        # General commits
        general_stmt = select(Commit).order_by(Commit.authored_date.desc()).limit(limit)
        general_commits = session.exec(general_stmt).all()
        
        # Merge uniquely
        all_commits_dict = {c.sha: c for c in (list(revert_commits) + list(general_commits))}
        
        # Also ensure commits linked from PRs and Issues are loaded
        needed_shas = set()
        for pr in prs[:50]:
            for s in (pr.linked_commit_shas or []):
                if len(s) == 40 and s not in all_commits_dict:
                    needed_shas.add(s)
        for iss in issues[:50]:
            for s in (iss.linked_commit_shas or []):
                if len(s) == 40 and s not in all_commits_dict:
                    needed_shas.add(s)
                    
        if needed_shas:
            extra_commits = session.exec(select(Commit).where(Commit.sha.in_(list(needed_shas)[:100]))).all()
            for c in extra_commits:
                all_commits_dict[c.sha] = c

        commits = list(all_commits_dict.values())

        # Populate Commit Nodes
        for c in commits:
            cid = f"commit:{c.sha[:7]}"
            if cid not in node_ids:
                node_ids.add(cid)
                is_rev = bool(c.is_revert or c.reverts_sha or c.superseded_by_sha)
                nodes.append({
                    "id": cid,
                    "type": "revert" if is_rev else "commit",
                    "label": f"commit {c.sha[:7]}",
                    "title": c.message.split("\n")[0] if c.message else "Commit",
                    "sha": c.sha,
                    "author": c.author_name or "unknown",
                    "date": c.authored_date.strftime("%Y-%m-%d") if c.authored_date else "",
                    "symbols": c.symbols_modified or [],
                    "files": c.files_changed or [],
                    "is_revert": c.is_revert,
                    "reverts_sha": c.reverts_sha,
                    "superseded_by_sha": c.superseded_by_sha,
                    "insertions": c.insertions,
                    "deletions": c.deletions
                })

        # 4. Fetch AST Symbols
        symbols = session.exec(select(SymbolIndex).order_by(SymbolIndex.commit_count.desc()).limit(60)).all()
        for sym in symbols:
            sym_id = f"symbol:{sym.symbol_name}"
            if sym_id not in node_ids:
                node_ids.add(sym_id)
                nodes.append({
                    "id": sym_id,
                    "type": "symbol",
                    "label": sym.symbol_name,
                    "kind": sym.kind or "symbol",
                    "file_path": sym.file_path,
                    "commits_count": sym.commit_count
                })

        # 5. Build Explicit Causal Edges
        # Issue -> PR ("RESOLVED_BY")
        for iss in issues:
            iss_id = f"issue:#{iss.number}"
            for pr_num in (iss.linked_pr_numbers or []):
                pr_id = f"pr:#{pr_num}"
                add_edge(iss_id, pr_id, "resolves", "RESOLVED_BY_PR")
                
        for pr in prs:
            pr_id = f"pr:#{pr.number}"
            # PR -> Issue
            for is_num in (pr.linked_issue_numbers or []):
                iss_id = f"issue:#{is_num}"
                add_edge(iss_id, pr_id, "resolves", "RESOLVES_ISSUE")
            
            # PR -> Commit ("MERGES")
            for sha in (pr.linked_commit_shas or []):
                c_id = f"commit:{sha[:7]}"
                add_edge(pr_id, c_id, "merges", "MERGED_IN_COMMIT")
        
        # Commit -> Revert Commit ("REVERTS")
        for c in commits:
            cid = f"commit:{c.sha[:7]}"
            if c.reverts_sha:
                target_id = f"commit:{c.reverts_sha[:7]}"
                add_edge(cid, target_id, "reverts", "REVERTS_COMMIT", highlight=True)
            if c.superseded_by_sha:
                target_id = f"commit:{c.superseded_by_sha[:7]}"
                add_edge(cid, target_id, "superseded_by", "SUPERSEDED_BY", highlight=True)
                
            # Commit -> Symbol ("MODIFIES")
            for sym_name in (c.symbols_modified or []):
                sym_id = f"symbol:{sym_name}"
                add_edge(cid, sym_id, "modifies", "MODIFIES_SYMBOL")

    return {
        "repo_id": repo_id,
        "repo_name": config["name"],
        "total_repo_commits": config.get("total_commits", len(commits)),
        "nodes_count": len(nodes),
        "edges_count": len(edges),
        "nodes": nodes,
        "edges": edges
    }

@app.post("/api/ask")
async def ask_question(payload: Dict[str, Any], x_gemini_api_key: Optional[str] = Header(None)):
    """Synchronous forensic query endpoint."""
    repo_id = payload.get("repo_id", "requests")
    question = payload.get("question")
    api_key = payload.get("api_key") or x_gemini_api_key
    
    if not question:
        raise HTTPException(status_code=400, detail="Missing required 'question' parameter.")
    
    _activate_repo_environment(repo_id, client_api_key=api_key)
    
    try:
        response = ask_tool(question=question, repo_id=repo_id)
        return {
            "repo_id": repo_id,
            "question": question,
            "response": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ask/stream")
async def stream_question(
    repo_id: str = Query("requests", description="Repository ID (requests, flask, mss)"),
    q: str = Query(..., description="The forensic causal question to investigate"),
    api_key: Optional[str] = Query(None, description="User Gemini API Key"),
    x_gemini_api_key: Optional[str] = Header(None)
):
    """Real-time Server-Sent Events (SSE) stream showing the LangGraph execution lifecycle."""
    client_key = api_key or x_gemini_api_key
    config = _activate_repo_environment(repo_id, client_api_key=client_key)
    
    async def event_generator():
        yield {
            "event": "start",
            "data": json.dumps({
                "repo_id": repo_id,
                "repo_name": config["name"],
                "question": q,
                "status": "Initializing Codebase History Analyzer state machine..."
            })
        }
        await asyncio.sleep(0.1)
        
        # Step 1: Candidate Forensics Discovery
        yield {
            "event": "planning",
            "data": json.dumps({
                "step": "Candidate Forensics",
                "message": f"Decomposing query across {config['name']} AST symbols, commit hashes, and PR cross-references..."
            })
        }
        await asyncio.sleep(0.2)
        
        # Run agent graph execution in background thread
        try:
            inputs = {
                "question": q,
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
            
            yield {
                "event": "retrieval",
                "data": json.dumps({
                    "step": "Hybrid RRF Retrieval",
                    "message": "Executing dense vector embeddings and lexical BM25 ranking over historical git diffs..."
                })
            }
            
            loop = asyncio.get_event_loop()
            final_state = await loop.run_in_executor(
                None,
                lambda: agent_graph.invoke(inputs, config={"recursion_limit": 60})
            )
            
            sub_q = final_state.get("sub_questions", [])
            retrieved_count = len(final_state.get("retrieved_chunks", []))
            verification = final_state.get("verification_passed", True)
            answer = final_state.get("response", "Could not synthesize answer.")
            
            yield {
                "event": "verification",
                "data": json.dumps({
                    "step": "Self-Verification Judge",
                    "sub_questions": sub_q,
                    "chunks_evaluated": retrieved_count,
                    "verification_passed": verification,
                    "message": f"Verified causal claims against {retrieved_count} repository history chunks."
                })
            }
            await asyncio.sleep(0.1)
            
            yield {
                "event": "answer",
                "data": json.dumps({
                    "response": answer,
                    "retrieved_chunks_count": retrieved_count,
                    "sub_questions": sub_q
                })
            }
            
            yield {
                "event": "done",
                "data": json.dumps({"status": "complete"})
            }
        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }

    return EventSourceResponse(event_generator())

@app.get("/api/hotspots/{repo_id}")
async def get_hotspots(repo_id: str, top_n: int = Query(15, ge=1, le=50)):
    """Returns top churn files ranked by historical commit count."""
    _activate_repo_environment(repo_id)
    hotspots = repo_hotspots_tool(top_n=top_n)
    return {"repo_id": repo_id, "hotspots": hotspots}

@app.get("/api/ownership/{repo_id}")
async def get_ownership(repo_id: str, file_path: Optional[str] = Query(None)):
    """Returns contributor ownership distribution and automated bus factor risk assessment."""
    _activate_repo_environment(repo_id)
    ownership = repo_ownership_tool(file_path=file_path)
    return {"repo_id": repo_id, "ownership": ownership}

@app.get("/api/coupling/{repo_id}")
async def get_coupling(repo_id: str, min_co_commits: int = Query(2, ge=1), top_n: int = Query(15, ge=1, le=50)):
    """Returns temporal file change coupling pairs (files that change together)."""
    _activate_repo_environment(repo_id)
    couplings = change_coupling_tool(min_co_commits=min_co_commits, top_n=top_n)
    return {"repo_id": repo_id, "couplings": couplings}

@app.get("/api/symbols/{repo_id}")
async def get_symbols(repo_id: str, top_n: int = Query(40, ge=1, le=100)):
    """Returns extracted AST code symbols ranked by modification frequency."""
    _activate_repo_environment(repo_id)
    symbols = repo_symbols_tool(top_n=top_n)
    return {"repo_id": repo_id, "symbols": symbols}

@app.get("/api/symbols/{repo_id}/history")
async def get_symbol_history(repo_id: str, symbol: str = Query(...)):
    """Traces all commits that modified a specific AST class or function."""
    _activate_repo_environment(repo_id)
    history = symbol_history_tool(symbol_query=symbol)
    # Ensure author_name and authored_date aliases
    for item in history:
        item["author_name"] = item.get("author") or item.get("author_name") or "contributor"
        item["authored_date"] = item.get("date") or item.get("authored_date") or ""
    return {"repo_id": repo_id, "symbol": symbol, "commits": history}

@app.get("/api/eval/leaderboard")
async def get_leaderboard():
    """Returns benchmark evaluations across all indexed repositories."""
    results = {}
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    for repo_key, res_file in [
        ("requests", "requests_results.json"),
        ("flask", "flask_results.json"),
        ("mss", "mss_results.json")
    ]:
        fpath = os.path.join(workspace_root, "eval", res_file)
        if os.path.exists(fpath):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    results[repo_key] = json.load(f)
            except Exception:
                pass
                
    return {"leaderboard": results}

# Mount static web UI: support both installed pip package and local dev repo
pkg_static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "static"))
dev_static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "web"))

if os.path.exists(pkg_static_dir) and os.path.exists(os.path.join(pkg_static_dir, "index.html")):
    app.mount("/", StaticFiles(directory=pkg_static_dir, html=True), name="static")
elif os.path.exists(dev_static_dir):
    app.mount("/", StaticFiles(directory=dev_static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("archaeologist.web.server:app", host="0.0.0.0", port=8000, reload=True)
