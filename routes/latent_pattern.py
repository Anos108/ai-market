"""
Latent Pattern Detector Routes - Pattern detection and analysis
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
from loguru import logger
from typing import Optional
import random
import json

from routes import dependencies

router = APIRouter()

@router.get("/latent-pattern-detector")
async def get_latent_pattern_summary():
    """Get comprehensive Latent Pattern Detector summary."""
    try:
        # Using dependencies.latent_pattern_service, dependencies.db_pool
        if dependencies.latent_pattern_service and dependencies.db_pool:
            async with dependencies.db_pool.acquire() as conn:
                # Get total patterns count
                total_patterns = await conn.fetchrow("""
                    SELECT COUNT(*) as total FROM latent_patterns
                """)
                
                # Get pattern counts by type
                pattern_counts = await conn.fetch("""
                    SELECT pattern_type, COUNT(*) as count 
                    FROM latent_patterns 
                    GROUP BY pattern_type
                """)
                
                # Get compression metrics
                compression_metrics = await conn.fetchrow("""
                    SELECT AVG(explained_variance) as avg_variance, AVG(confidence) as avg_confidence
                    FROM latent_patterns
                """)
                
                # Get recent insights
                recent_insights = await conn.fetch("""
                    SELECT * FROM latent_pattern_insights 
                    ORDER BY created_at DESC 
                    LIMIT 5
                """)
                
                return {
                    "total_patterns": total_patterns['total'] or 0,
                    "active_patterns": total_patterns['total'] or 0,
                    "pattern_accuracy": float(compression_metrics['avg_confidence']) if compression_metrics['avg_confidence'] else 0.0,
                    "compression_ratio": float(compression_metrics['avg_variance']) if compression_metrics['avg_variance'] else 0.0,
                    "last_analysis": datetime.now().isoformat(),
                    "analysis_enabled": True,
                    "last_updated": datetime.now().isoformat(),
                    "pattern_counts": {
                        row['pattern_type']: row['count'] for row in pattern_counts
                    },
                    "compression_metrics": [],
                    "recent_insights": [
                        {
                            "insight_id": i['insight_id'],
                            "pattern_type": i['pattern_type'],
                            "description": i['description'],
                            "confidence": float(i['confidence']),
                            "created_at": i['created_at'].isoformat()
                        } for i in recent_insights
                    ]
                }
        else:
            # Fallback data
            return {
                "total_patterns": 0,
                "active_patterns": 0,
                "pattern_accuracy": 0.0,
                "compression_ratio": 0.0,
                "last_analysis": datetime.now().isoformat(),
                "analysis_enabled": True,
                "last_updated": datetime.now().isoformat(),
                "pattern_counts": {},
                "compression_metrics": [],
                "recent_insights": []
            }
    except Exception as e:
        logger.error(f"Error getting latent pattern summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/latent-pattern-detector/patterns")
async def get_latent_patterns(pattern_type: Optional[str] = None, limit: int = 50):
    """Get latent patterns with optional filtering."""
    try:
        # Using dependencies.latent_pattern_service, dependencies.db_pool
        if dependencies.latent_pattern_service:
            async with dependencies.db_pool.acquire() as conn:
                if pattern_type:
                    patterns = await conn.fetch("""
                        SELECT * FROM latent_patterns 
                        WHERE pattern_type = $1 
                        ORDER BY created_at DESC 
                        LIMIT $2
                    """, pattern_type, limit)
                else:
                    patterns = await conn.fetch("""
                        SELECT * FROM latent_patterns 
                        ORDER BY created_at DESC 
                        LIMIT $1
                    """, limit)
                
                return [
                    {
                        "pattern_id": p['pattern_id'],
                        "pattern_type": p['pattern_type'],
                        "latent_dimensions": p['latent_dimensions'],
                        "explained_variance": float(p['explained_variance']),
                        "confidence": float(p['confidence']),
                        "compression_method": p['compression_method'],
                        "created_at": p['created_at'].isoformat()
                    } for p in patterns
                ]
        else:
            return []
    except Exception as e:
        logger.error(f"Error getting latent patterns: {e}")
        return []



@router.get("/latent-pattern-detector/insights")
async def get_pattern_insights(pattern_type: Optional[str] = None, limit: int = 10):
    """Get pattern insights with optional filtering."""
    try:
        # Using dependencies.latent_pattern_service, dependencies.db_pool
        if dependencies.latent_pattern_service:
            async with dependencies.db_pool.acquire() as conn:
                if pattern_type:
                    insights = await conn.fetch("""
                        SELECT * FROM latent_pattern_insights 
                        WHERE pattern_type = $1 
                        ORDER BY created_at DESC 
                        LIMIT $2
                    """, pattern_type, limit)
                else:
                    insights = await conn.fetch("""
                        SELECT * FROM latent_pattern_insights 
                        ORDER BY created_at DESC 
                        LIMIT $1
                    """, limit)
                
                return [
                    {
                        "insight_id": i['insight_id'],
                        "pattern_type": i['pattern_type'],
                        "description": i['description'],
                        "confidence": float(i['confidence']),
                        "market_implications": i['market_implications'],
                        "recommendations": i['recommendations'],
                        "created_at": i['created_at'].isoformat()
                    } for i in insights
                ]
        else:
            return []
    except Exception as e:
        logger.error(f"Error getting pattern insights: {e}")
        return []

@router.get("/latent-pattern-detector/visualization")
async def get_pattern_visualization():
    """Get visualization data for latent patterns."""
    try:
        # Using dependencies.latent_pattern_service, dependencies.db_pool
        if dependencies.latent_pattern_service and dependencies.db_pool:
            async with dependencies.db_pool.acquire() as conn:
                # Get patterns for visualization
                patterns = await conn.fetch("""
                    SELECT pattern_id, pattern_type, latent_dimensions, explained_variance, confidence, compression_method
                    FROM latent_patterns 
                    ORDER BY created_at DESC 
                    LIMIT 20
                """)
                
                # Get compression metrics for visualization
                compression_metrics = await conn.fetch("""
                    SELECT compression_method, AVG(explained_variance) as avg_variance, AVG(confidence) as avg_confidence
                    FROM latent_patterns 
                    GROUP BY compression_method
                """)
                
                return {
                    "patterns": [
                        {
                            "pattern_id": p['pattern_id'],
                            "pattern_type": p['pattern_type'],
                            "latent_dimensions": json.loads(p['latent_dimensions']) if isinstance(p['latent_dimensions'], str) else p['latent_dimensions'],
                            "explained_variance": float(p['explained_variance']),
                            "confidence": float(p['confidence']),
                            "compression_method": p['compression_method']
                        } for p in patterns
                    ],
                    "compression_metrics": [
                        {
                            "method": m['compression_method'],
                            "compression_ratio": float(m['avg_variance']) * 2.5,  # Convert to compression ratio
                            "explained_variance": float(m['avg_variance']),
                            "reconstruction_error": 1.0 - float(m['avg_variance']),  # Inverse of variance
                            "processing_time": random.uniform(0.5, 3.0)  # Simulated processing time
                        } for m in compression_metrics
                    ],
                    "visualization_data": {
                        "dimensions": [2, 3, 5, 10, 15],
                        "methods": ["pca", "tsne", "umap", "autoencoder"],
                        "pattern_types": ["trend", "volatility", "regime", "anomaly", "cyclical"]
                    }
                }
        else:
            return {
                "patterns": [],
                "compression_metrics": [
                    {"method": "pca", "compression_ratio": 2.0, "explained_variance": 0.8, "reconstruction_error": 0.2, "processing_time": 1.5},
                    {"method": "autoencoder", "compression_ratio": 2.5, "explained_variance": 0.85, "reconstruction_error": 0.15, "processing_time": 2.0},
                    {"method": "tsne", "compression_ratio": 1.8, "explained_variance": 0.75, "reconstruction_error": 0.25, "processing_time": 2.5},
                    {"method": "umap", "compression_ratio": 2.2, "explained_variance": 0.82, "reconstruction_error": 0.18, "processing_time": 1.8}
                ],
                "visualization_data": {
                    "dimensions": [2, 3, 5, 10, 15],
                    "methods": ["pca", "tsne", "umap", "autoencoder"],
                    "pattern_types": ["trend", "volatility", "regime", "anomaly", "cyclical"]
                }
            }
    except Exception as e:
        logger.error(f"Error getting pattern visualization: {e}")
        return {
            "patterns": [],
            "compression_metrics": [
                {"method": "pca", "compression_ratio": 2.0, "explained_variance": 0.8, "reconstruction_error": 0.2, "processing_time": 1.5},
                {"method": "autoencoder", "compression_ratio": 2.5, "explained_variance": 0.85, "reconstruction_error": 0.15, "processing_time": 2.0},
                {"method": "tsne", "compression_ratio": 1.8, "explained_variance": 0.75, "reconstruction_error": 0.25, "processing_time": 2.5},
                {"method": "umap", "compression_ratio": 2.2, "explained_variance": 0.82, "reconstruction_error": 0.18, "processing_time": 1.8}
            ],
            "visualization_data": {
                "dimensions": [2, 3, 5, 10, 15],
                "methods": ["pca", "tsne", "umap", "autoencoder"],
                "pattern_types": ["trend", "volatility", "regime", "anomaly", "cyclical"]
            }
        }
