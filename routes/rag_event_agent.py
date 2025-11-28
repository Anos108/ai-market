"""
RAG Event Agent Routes - LLM-RAG powered event analysis
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
from loguru import logger
from typing import Optional
import random
import json

from routes import dependencies

router = APIRouter()


@router.get("/rag-event-agent/summary")
async def get_rag_summary():
    """Get RAG Event Agent summary with real data."""
    try:
        # Using dependencies.rag_event_agent_service, dependencies.db_pool
        # Initialize service if not already done
        if not dependencies.rag_event_agent_service and dependencies.db_pool:
            from services.rag_event_agent_service import RAGEventAgentService
            dependencies.rag_event_agent_service = RAGEventAgentService(dependencies.db_pool)
            # Create sample data if needed
            await dependencies.rag_event_agent_service.create_sample_data()
        
        if dependencies.rag_event_agent_service:
            return await dependencies.rag_event_agent_service.get_rag_event_agent_summary()
        else:
            # Fallback summary
            return {
                "total_documents": 0,
                "vector_db_size": 0,
                "last_news_update": datetime.now().isoformat(),
                "rag_accuracy": 0.0,
                "llm_enabled": True,
                "active_sources": 0,
                "total_queries": 0,
                "avg_response_time": 0,
                "avg_confidence": 0.0,
                "last_updated": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Error fetching RAG summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag-event-agent/analysis")
async def get_rag_analysis():
    """Get latest RAG analysis with real Ollama LLM."""
    try:
        # Using dependencies.rag_event_agent_service, dependencies.db_pool
        # Initialize service if not already done
        if not dependencies.rag_event_agent_service and dependencies.db_pool:
            from services.rag_event_agent_service import RAGEventAgentService
            dependencies.rag_event_agent_service = RAGEventAgentService(dependencies.db_pool)
            # Create sample data if needed
            await dependencies.rag_event_agent_service.create_sample_data()
        
        if dependencies.rag_event_agent_service:
            return await dependencies.rag_event_agent_service.get_rag_analysis()
        else:
            # Fallback analysis
            return {
                "query": "No analysis available",
                "relevant_docs": [],
                "llm_response": "No LLM response available",
                "confidence": 0.0,
                "reasoning": "No reasoning available",
                "analysis_type": "none",
                "response_time_ms": 0,
                "created_at": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Error fetching RAG analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag-event-agent/analyze")
async def analyze_with_rag(request: dict):
    """Analyze a custom query using RAG with Ollama LLM."""
    try:
        query = request.get("query", "")
        sector = request.get("sector", None)
        
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query is required")
        
        # Using dependencies.rag_event_agent_service, dependencies.db_pool
        # Initialize service if not already done
        if not dependencies.rag_event_agent_service and dependencies.db_pool:
            from services.rag_event_agent_service import RAGEventAgentService
            dependencies.rag_event_agent_service = RAGEventAgentService(dependencies.db_pool)
            # Create sample data if needed
            await dependencies.rag_event_agent_service.create_sample_data()
        
        if dependencies.rag_event_agent_service:
            return await dependencies.rag_event_agent_service.get_rag_analysis(query.strip(), sector)
        else:
            raise HTTPException(status_code=500, detail="RAG service not available")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in RAG analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag-event-agent/sector-analysis/{sector}")
async def get_sector_analysis(sector: str):
    """Get sector-specific RAG analysis."""
    try:
        # Validate sector
        valid_sectors = ['technology', 'finance', 'healthcare', 'retail']
        if sector.lower() not in valid_sectors:
            raise HTTPException(status_code=400, detail=f"Invalid sector. Must be one of: {valid_sectors}")
        
        # Initialize service if not already done
        if not dependencies.rag_event_agent_service and dependencies.db_pool:
            from services.rag_event_agent_service import RAGEventAgentService
            dependencies.rag_event_agent_service = RAGEventAgentService(dependencies.db_pool)
            # Create sample data if needed
            await dependencies.rag_event_agent_service.create_sample_data()
        
        if dependencies.rag_event_agent_service:
            return await dependencies.rag_event_agent_service.get_sector_analysis(sector.lower())
        else:
            raise HTTPException(status_code=500, detail="RAG service not available")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in sector analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag-event-agent/multi-sector-analysis")
async def get_multi_sector_analysis():
    """Get analysis for all major sectors (technology, finance, healthcare, retail)."""
    try:
        # Initialize service if not already done
        if not dependencies.rag_event_agent_service and dependencies.db_pool:
            from services.rag_event_agent_service import RAGEventAgentService
            dependencies.rag_event_agent_service = RAGEventAgentService(dependencies.db_pool)
            # Create sample data if needed
            await dependencies.rag_event_agent_service.create_sample_data()
        
        if dependencies.rag_event_agent_service:
            return await dependencies.rag_event_agent_service.get_multi_sector_analysis()
        else:
            raise HTTPException(status_code=500, detail="RAG service not available")
            
    except Exception as e:
        logger.error(f"Error in multi-sector analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag-event-agent/automation-status")
async def get_automation_status():
    """Get the status of the automated RAG service."""
    try:
        if dependencies.automated_rag_service:
            status = await dependencies.automated_rag_service.get_status()
            return status
        else:
            return {
                "is_running": False,
                "message": "Automated RAG service not initialized"
            }
    except Exception as e:
        logger.error(f"Error getting automation status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag-event-agent/force-update")
async def force_rag_update():
    """Force an immediate RAG update cycle (fetch news + generate analyses)."""
    try:
        if dependencies.automated_rag_service:
            result = await dependencies.automated_rag_service.force_update()
            return result
        else:
            raise HTTPException(status_code=500, detail="Automated RAG service not initialized")
    except Exception as e:
        logger.error(f"Error forcing RAG update: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag-event-agent/latest-analysis")
async def get_latest_rag_analysis():
    """Get the latest RAG analysis from database for all sectors."""
    try:
        # Initialize service if not already done
        if not dependencies.rag_event_agent_service and dependencies.db_pool:
            from services.rag_event_agent_service import RAGEventAgentService
            dependencies.rag_event_agent_service = RAGEventAgentService(dependencies.db_pool)
            # Create sample data if needed
            await dependencies.rag_event_agent_service.create_sample_data()
        
        if dependencies.rag_event_agent_service and dependencies.db_pool:
            async with dependencies.db_pool.acquire() as conn:
                # Get latest analysis for each sector based on analysis_type
                results = await conn.fetch("""
                    SELECT DISTINCT ON (analysis_type)
                        analysis_type,
                        query,
                        llm_response,
                        confidence,
                        created_at,
                        response_time_ms,
                        relevant_doc_ids
                    FROM rag_analysis 
                    WHERE analysis_type IN ('technology_impact', 'finance_impact', 'healthcare_impact', 'retail_impact')
                    AND created_at >= NOW() - INTERVAL '24 hours'
                    ORDER BY analysis_type, created_at DESC
                """)
                
                sector_analyses = {}
                for row in results:
                    # Extract sector from analysis_type (e.g., 'technology_impact' -> 'technology')
                    sector = row['analysis_type'].replace('_impact', '')
                    
                    # Count sources from relevant_doc_ids
                    source_count = len(row.get('relevant_doc_ids', [])) if row.get('relevant_doc_ids') else 0
                    
                    sector_analyses[sector] = {
                        "query": row['query'],
                        "llm_response": row['llm_response'],
                        "confidence": float(row['confidence']),
                        "sources": f"{source_count} news sources",
                        "source_count": source_count,
                        "created_at": row['created_at'].isoformat(),
                        "response_time_ms": row['response_time_ms']
                    }
                
                return {
                    "success": True,
                    "sector_analyses": sector_analyses,
                    "last_updated": datetime.now().isoformat()
                }
        else:
            raise HTTPException(status_code=500, detail="RAG service not available")
            
    except Exception as e:
        logger.error(f"Error getting latest RAG analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag-event-agent/documents")
async def get_rag_documents(limit: int = 20):
    """Get recent news documents from RAG Event Agent with real data."""
    try:
        # Using dependencies.rag_event_agent_service, dependencies.db_pool
        # Initialize service if not already done
        if not dependencies.rag_event_agent_service and dependencies.db_pool:
            from services.rag_event_agent_service import RAGEventAgentService
            dependencies.rag_event_agent_service = RAGEventAgentService(dependencies.db_pool)
            # Create sample data if needed
            await dependencies.rag_event_agent_service.create_sample_data()
        
        if dependencies.rag_event_agent_service:
            return await dependencies.rag_event_agent_service.get_rag_documents(limit)
        else:
            # Fallback documents
            return []
    except Exception as e:
        logger.error(f"Error fetching RAG documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/rag-event-agent/refresh-news")
async def refresh_news_articles():
    """Refresh news articles by fetching latest from external sources."""
    try:
        # Using dependencies.rag_event_agent_service, dependencies.db_pool
        # Initialize service if not already done
        if not dependencies.rag_event_agent_service and dependencies.db_pool:
            from services.rag_event_agent_service import RAGEventAgentService
            dependencies.rag_event_agent_service = RAGEventAgentService(dependencies.db_pool)
        
        if dependencies.rag_event_agent_service:
            # Clear existing news documents
            async with dependencies.db_pool.acquire() as conn:
                await conn.execute("DELETE FROM rag_news_documents")
            
            # Fetch and store new real news articles
            await dependencies.rag_event_agent_service.create_sample_data()
            
            return {
                "status": "success",
                "message": "News articles refreshed successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="RAG service not available")
            
    except Exception as e:
        logger.error(f"Error refreshing news articles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag-event-agent/performance")
async def get_rag_performance():
    """Get RAG system performance metrics from RAG Event Agent with real data."""
    try:
        # Using dependencies.rag_event_agent_service, dependencies.db_pool
        # Initialize service if not already done
        if not dependencies.rag_event_agent_service and dependencies.db_pool:
            from services.rag_event_agent_service import RAGEventAgentService
            dependencies.rag_event_agent_service = RAGEventAgentService(dependencies.db_pool)
            # Create sample data if needed
            await dependencies.rag_event_agent_service.create_sample_data()
        
        if dependencies.rag_event_agent_service:
            return await dependencies.rag_event_agent_service.get_rag_performance()
        else:
            # Fallback performance
            return {"metrics": {}, "last_updated": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error fetching RAG performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rag-event-agent/historical-analyses")
async def get_historical_analyses(limit: int = 20, days: int = 7):
    """Get historical RAG analyses for insights and trend analysis."""
    try:
        # Initialize service if not already done
        if not dependencies.rag_event_agent_service and dependencies.db_pool:
            from services.rag_event_agent_service import RAGEventAgentService
            dependencies.rag_event_agent_service = RAGEventAgentService(dependencies.db_pool)
        
        if dependencies.rag_event_agent_service:
            return await dependencies.rag_event_agent_service.get_historical_analyses(limit, days)
        else:
            return []
    except Exception as e:
        logger.error(f"Error fetching historical analyses: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag-event-agent/analysis-insights")
async def get_analysis_insights(days: int = 30):
    """Get comprehensive insights from historical RAG analyses."""
    try:
        # Initialize service if not already done
        if not dependencies.rag_event_agent_service and dependencies.db_pool:
            from services.rag_event_agent_service import RAGEventAgentService
            dependencies.rag_event_agent_service = RAGEventAgentService(dependencies.db_pool)
        
        if dependencies.rag_event_agent_service:
            return await dependencies.rag_event_agent_service.get_analysis_insights(days)
        else:
            return {
                "analysis_statistics": {"total_analyses": 0, "avg_confidence": 0.0},
                "query_patterns": [],
                "confidence_trends": [],
                "performance_metrics": {},
                "insights_generated_at": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Error fetching analysis insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag-event-agent/analysis/{analysis_id}")
async def get_specific_analysis(analysis_id: str):
    """Get a specific RAG analysis by ID."""
    try:
        if not dependencies.db_pool:
            raise HTTPException(status_code=500, detail="Database not available")
        
        # Extract the numeric ID from the analysis_id format (analysis_123_20241011_123456)
        try:
            numeric_id = int(analysis_id.split('_')[1])
        except (IndexError, ValueError):
            raise HTTPException(status_code=400, detail="Invalid analysis ID format")
        
        async with dependencies.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT 
                    id,
                    query,
                    llm_response,
                    confidence,
                    reasoning,
                    analysis_type,
                    relevant_doc_ids,
                    response_time_ms,
                    created_at
                FROM rag_analysis
                WHERE id = $1
            """, numeric_id)
            
            if not row:
                raise HTTPException(status_code=404, detail="Analysis not found")
            
            formatted_analysis_id = f"analysis_{row['id']}_{row['created_at'].strftime('%Y%m%d_%H%M%S')}"
            
            return {
                "analysis_id": formatted_analysis_id,
                "query": row['query'],
                "llm_response": row['llm_response'],
                "confidence": float(row['confidence']),
                "reasoning": row['reasoning'],
                "analysis_type": row['analysis_type'],
                "relevant_doc_ids": row['relevant_doc_ids'],
                "response_time_ms": row['response_time_ms'],
                "created_at": row['created_at'].isoformat()
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching specific analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag-event-agent/sources")
async def get_analysis_sources(days: int = 30):
    """Get analysis sources and their usage statistics."""
    try:
        if not dependencies.db_pool:
            raise HTTPException(status_code=500, detail="Database not available")
        
        async with dependencies.db_pool.acquire() as conn:
            # Get source usage statistics from recent analyses
            source_stats = await conn.fetch("""
                SELECT 
                    unnest(relevant_doc_ids) as doc_id,
                    COUNT(*) as usage_count
                FROM rag_analysis
                WHERE created_at >= NOW() - INTERVAL '%s days'
                GROUP BY doc_id
                ORDER BY usage_count DESC
                LIMIT 20
            """ % days)
            
            # Get document details for these sources
            doc_ids = [row['doc_id'] for row in source_stats]
            doc_details = {}
            if doc_ids:
                doc_rows = await conn.fetch("""
                    SELECT doc_id, title, source, category, published_at
                    FROM rag_news_documents
                    WHERE doc_id = ANY($1)
                """, doc_ids)
                
                for doc_row in doc_rows:
                    doc_details[doc_row['doc_id']] = {
                        "title": doc_row['title'],
                        "source": doc_row['source'],
                        "category": doc_row['category'],
                        "published_at": doc_row['published_at'].isoformat()
                    }
            
            # Combine statistics with document details
            sources = []
            for row in source_stats:
                doc_id = row['doc_id']
                if doc_id in doc_details:
                    sources.append({
                        "doc_id": doc_id,
                        "usage_count": row['usage_count'],
                        **doc_details[doc_id]
                    })
            
            return {
                "sources": sources,
                "total_unique_sources": len(sources),
                "analysis_period_days": days,
                "generated_at": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error fetching analysis sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# RL Strategy Agent endpoints for reinforcement learning strategy optimization