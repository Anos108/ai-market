"""
Ticker Discovery Routes - Market scanning and ticker discovery
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
from loguru import logger
from typing import Optional
import random
import json

from routes import dependencies
from routes.utils import run_automated_ticker_discovery

router = APIRouter()


@router.get("/ticker-discovery/scanner-summary")
async def get_scanner_summary():
    """Get ticker discovery scanner summary."""
    try:
        # Using dependencies.db_pool
        if dependencies.db_pool:
            async with dependencies.db_pool.acquire() as conn:
                # Get data from the real ticker discovery tables
                result = await conn.fetchrow("""
                    SELECT 
                        COALESCE(SUM(tdh.total_scanned), 0) as total_scanned,
                        COALESCE(SUM(tdh.triggers_found), 0) as triggers_found,
                        COALESCE(SUM(tdh.high_priority), 0) as high_priority,
                        COALESCE(AVG(tdr.confidence), 0) as avg_confidence
                    FROM ticker_discovery_history tdh
                    LEFT JOIN ticker_discovery_results tdr ON tdh.scan_id = tdr.scan_id
                    WHERE tdh.scan_timestamp >= NOW() - INTERVAL '24 hours'
                """)
                
                if result:
                    return {
                        "total_scanned": result['total_scanned'] or 0,
                        "triggers_found": result['triggers_found'] or 0,
                        "high_priority": result['high_priority'] or 0,
                        "avg_confidence": float(result['avg_confidence'] or 0),
                        "last_updated": datetime.now().isoformat()
                    }
        
        # Fallback if no data found
        return {
            "total_scanned": 0,
            "triggers_found": 0,
            "high_priority": 0,
            "avg_confidence": 0.0,
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting scanner summary: {e}")
        return {
            "total_scanned": 0,
            "triggers_found": 0,
            "high_priority": 0,
            "avg_confidence": 0.0,
            "last_updated": datetime.now().isoformat()
        }



@router.get("/ticker-discovery/ranker-summary")
async def get_ranker_summary():
    """Get ticker discovery ranker summary."""
    try:
        # Using dependencies.db_pool
        if dependencies.db_pool:
            async with dependencies.db_pool.acquire() as conn:
                result = await conn.fetchrow("""
                    SELECT 
                        COALESCE(COUNT(tdr.symbol), 0) as total_ranked,
                        COALESCE(AVG(tdr.score), 0) as avg_score,
                        COALESCE(AVG(tdr.confidence), 0) as avg_confidence
                    FROM ticker_discovery_results tdr
                    JOIN ticker_discovery_history tdh ON tdr.scan_id = tdh.scan_id
                    WHERE tdh.scan_timestamp >= NOW() - INTERVAL '24 hours'
                """)
                
                if result:
                    return {
                        "total_ranked": result['total_ranked'] or 0,
                        "avg_score": float(result['avg_score'] or 0),
                        "avg_confidence": float(result['avg_confidence'] or 0),
                        "last_updated": datetime.now().isoformat()
                    }

        # Fallback if no data found
        return {
            "total_ranked": 0,
            "avg_score": 0.0,
            "avg_confidence": 0.0,
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting ranker summary: {e}")
        return {
            "total_ranked": 0,
            "avg_score": 0.0,
            "avg_confidence": 0.0,
            "last_updated": datetime.now().isoformat()
        }



@router.get("/ticker-discovery/scan-details")
async def get_scan_details():
    """Get detailed scan results."""
    try:
        # Using dependencies.db_pool
        if dependencies.db_pool:
            async with dependencies.db_pool.acquire() as conn:
                # Get data from the latest scan only to avoid duplicates
                results = await conn.fetch("""
                    SELECT 
                        tdr.symbol, tdr.trigger_type, tdr.priority, tdr.confidence, 
                        tdr.score, tdr.description, tdr.sector, tdr.industry, 
                        tdh.scan_timestamp as created_at
                    FROM ticker_discovery_results tdr
                    JOIN ticker_discovery_history tdh ON tdr.scan_id = tdh.scan_id
                    WHERE tdh.scan_id = (
                        SELECT scan_id FROM ticker_discovery_history 
                        WHERE scan_timestamp >= NOW() - INTERVAL '24 hours'
                        ORDER BY scan_timestamp DESC LIMIT 1
                    )
                    ORDER BY tdr.score DESC, tdr.confidence DESC
                    LIMIT 50
                """)
                
                discovered_tickers = []
                for row in results:
                    discovered_tickers.append({
                        "symbol": row['symbol'],
                        "trigger": row['trigger_type'],
                        "priority": row['priority'],
                        "confidence": float(row['confidence']),
                        "score": float(row['score']),
                        "description": row['description'],
                        "sector": row['sector'],
                        "industry": row['industry'],
                        "timestamp": row['created_at'].isoformat()
                    })
                
                return {
                    "discovered_tickers": discovered_tickers,
                    "total_found": len(discovered_tickers),
                    "last_updated": datetime.now().isoformat()
                }
        
        # Fallback if no data found
        return {
            "discovered_tickers": [],
            "total_found": 0,
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting scan details: {e}")
        return {
            "discovered_tickers": [],
            "total_found": 0,
            "last_updated": datetime.now().isoformat()
        }


@router.post("/ticker-discovery/scan-market")
async def scan_market():
    """Trigger a market scan for ticker discovery."""
    try:
        logger.info("🚀 Starting market scan for ticker discovery...")
        
        # For now, return a simple success response
        return {
            "success": True,
            "message": "Market scan endpoint is working",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error during market scan: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to scan market: {str(e)}")


@router.post("/ticker-discovery/scan-sector/{sector}")
async def scan_sector(sector: str):
    """Scan a specific sector for market opportunities."""
    try:
        # Validate sector
        valid_sectors = ['technology', 'finance', 'healthcare', 'retail']
        if sector.lower() not in valid_sectors:
            raise HTTPException(status_code=400, detail=f"Invalid sector. Must be one of: {valid_sectors}")
        
        logger.info(f"🚀 Starting sector scan for {sector}...")
        
        # Using dependencies.db_pool
        if dependencies.db_pool:
            # Import the ticker scanner agent
            from agents.ticker_scanner_agent import TickerScannerAgent
            from services.real_data_service import RealDataService, RealDataConfig
            
            # Initialize services
            config = RealDataConfig()
            data_service = RealDataService(config)
            scanner_agent = TickerScannerAgent(data_service)
            
            # Scan the specific sector
            sector_results = await scanner_agent.scan_sector_opportunities(sector.lower())
            
            # Store results in database
            import uuid
            scan_id = str(uuid.uuid4())
            
            async with dependencies.db_pool.acquire() as conn:
                # Insert scan record
                await conn.execute("""
                    INSERT INTO ticker_discovery_history 
                    (scan_id, scan_timestamp, total_scanned, triggers_found, high_priority, notes)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """, scan_id, datetime.now(), len(sector_results), len(sector_results), 
                     len([r for r in sector_results if r.priority.value.upper() == 'HIGH']), f"sector_{sector}")
                
                # Insert individual results
                for result in sector_results:
                    await conn.execute("""
                        INSERT INTO ticker_discovery_results 
                        (scan_id, symbol, trigger_type, priority, confidence, score, 
                         description, discovered_at, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """, scan_id, result.symbol, result.trigger.value, result.priority.value.upper(),
                         result.confidence, result.score, result.description,
                         result.timestamp, json.dumps(result.metadata))
            
            return {
                "success": True,
                "message": f"Sector scan completed successfully for {sector}",
                "scan_id": scan_id,
                "sector": sector,
                "opportunities_found": len(sector_results),
                "high_priority": len([r for r in sector_results if r.priority.value.upper() == 'HIGH']),
                "scan_timestamp": datetime.now().isoformat()
            }
        
        return {
            "success": False,
            "message": "Database not available",
            "scan_timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in sector scan for {sector}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ticker-discovery/scan-all-sectors")
async def scan_all_sectors():
    """Scan all major sectors for market opportunities."""
    try:
        logger.info("🚀 Starting multi-sector scan...")
        
        # Using dependencies.db_pool
        if dependencies.db_pool:
            # Import the ticker scanner agent
            from agents.ticker_scanner_agent import TickerScannerAgent
            from services.real_data_service import RealDataService, RealDataConfig
            
            # Initialize services
            config = RealDataConfig()
            data_service = RealDataService(config)
            scanner_agent = TickerScannerAgent(data_service)
            
            # Scan all sectors
            all_sector_results = await scanner_agent.scan_all_sectors()
            
            # Store results in database
            import uuid
            scan_id = str(uuid.uuid4())
            total_opportunities = 0
            total_high_priority = 0
            
            async with dependencies.db_pool.acquire() as conn:
                for sector, sector_results in all_sector_results.items():
                    total_opportunities += len(sector_results)
                    total_high_priority += len([r for r in sector_results if r.priority.value.upper() == 'HIGH'])
                    
                    # Insert individual results for each sector
                    for result in sector_results:
                        await conn.execute("""
                            INSERT INTO ticker_discovery_results 
                            (scan_id, symbol, trigger_type, priority, confidence, score, 
                             description, discovered_at, metadata)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        """, scan_id, result.symbol, result.trigger.value, result.priority.value.upper(),
                         result.confidence, result.score, result.description,
                         result.timestamp, json.dumps(result.metadata))
                
                # Insert overall scan record
                await conn.execute("""
                    INSERT INTO ticker_discovery_history 
                    (scan_id, scan_timestamp, total_scanned, triggers_found, high_priority, notes)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """, scan_id, datetime.now(), total_opportunities, total_opportunities, 
                     total_high_priority, "multi_sector")
            
            # Prepare sector breakdown
            sector_breakdown = {}
            for sector, results in all_sector_results.items():
                sector_breakdown[sector] = {
                    "opportunities_found": len(results),
                    "high_priority": len([r for r in results if r.priority.value.upper() == 'HIGH']),
                    "avg_confidence": sum(r.confidence for r in results) / len(results) if results else 0
                }
            
            return {
                "success": True,
                "message": "Multi-sector scan completed successfully",
                "scan_id": scan_id,
                "total_opportunities": total_opportunities,
                "total_high_priority": total_high_priority,
                "sector_breakdown": sector_breakdown,
                "scan_timestamp": datetime.now().isoformat()
            }
        
        return {
            "success": False,
            "message": "Database not available",
            "scan_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in multi-sector scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ticker-discovery/sector-opportunities/{sector}")
async def get_sector_opportunities(sector: str, limit: int = 20):
    """Get recent opportunities for a specific sector."""
    try:
        # Validate sector
        valid_sectors = ['technology', 'finance', 'healthcare', 'retail']
        if sector.lower() not in valid_sectors:
            raise HTTPException(status_code=400, detail=f"Invalid sector. Must be one of: {valid_sectors}")
        
        if dependencies.db_pool:
            async with dependencies.db_pool.acquire() as conn:
                # Get recent opportunities for the sector
                results = await conn.fetch("""
                    SELECT 
                        tdr.symbol,
                        tdr.trigger_type,
                        tdr.priority,
                        tdr.confidence,
                        tdr.score,
                        tdr.description,
                        tdr.discovered_at,
                        tdr.metadata
                    FROM ticker_discovery_results tdr
                    JOIN ticker_discovery_history tdh ON tdr.scan_id = tdh.scan_id
                    WHERE tdh.notes = $1 
                    AND tdr.discovered_at >= NOW() - INTERVAL '24 hours'
                    ORDER BY tdr.score DESC, tdr.discovered_at DESC
                    LIMIT $2
                """, f"sector_{sector.lower()}", limit)
                
                opportunities = []
                for row in results:
                    opportunities.append({
                        "symbol": row['symbol'],
                        "trigger_type": row['trigger_type'],
                        "priority": row['priority'],
                        "confidence": float(row['confidence']),
                        "score": float(row['score']),
                        "description": row['description'],
                        "timestamp": row['discovered_at'].isoformat(),
                        "metadata": json.loads(row['metadata']) if row['metadata'] else {}
                    })
                
                return {
                    "sector": sector,
                    "opportunities": opportunities,
                    "total_found": len(opportunities),
                    "last_updated": datetime.now().isoformat()
                }
        
        return {
            "sector": sector,
            "opportunities": [],
            "total_found": 0,
            "message": "Database not available"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting sector opportunities for {sector}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
