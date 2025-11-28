"""
Data Quality Validation and Enhancement for AI Market Analysis System

This module provides comprehensive data quality validation, anomaly detection,
missing data handling, and data lineage tracking for market data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import warnings
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import json

logger = logging.getLogger(__name__)


class DataQualityLevel(Enum):
    """Data quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class DataQualityReport:
    """Data quality assessment report."""
    symbol: str
    source: str
    timestamp: datetime
    quality_level: DataQualityLevel
    completeness_score: float
    accuracy_score: float
    consistency_score: float
    timeliness_score: float
    anomalies_detected: int
    missing_data_points: int
    data_gaps: List[Tuple[datetime, datetime]]
    quality_issues: List[str]
    recommendations: List[str]


class DataQualityValidator:
    """Comprehensive data quality validation and enhancement."""
    
    def __init__(self):
        self.quality_thresholds = {
            'completeness': 0.95,  # 95% data completeness
            'accuracy': 0.90,      # 90% accuracy
            'consistency': 0.85,   # 85% consistency
            'timeliness': 0.80     # 80% timeliness
        }
        
    def validate_price_data(self, df: pd.DataFrame, symbol: str = None) -> DataQualityReport:
        """Comprehensive price data validation."""
        if df.empty:
            return self._create_empty_report(symbol or "unknown", "unknown")
        
        # Calculate quality scores
        completeness_score = self._calculate_completeness_score(df)
        accuracy_score = self._calculate_accuracy_score(df)
        consistency_score = self._calculate_consistency_score(df)
        timeliness_score = self._calculate_timeliness_score(df)
        
        # Detect anomalies
        anomalies = self._detect_anomalies(df)
        
        # Find missing data
        missing_data = self._find_missing_data(df)
        
        # Find data gaps
        data_gaps = self._find_data_gaps(df)
        
        # Identify quality issues
        quality_issues = self._identify_quality_issues(df, completeness_score, 
                                                     accuracy_score, consistency_score, timeliness_score)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(quality_issues, anomalies, missing_data)
        
        # Determine overall quality level
        overall_score = (completeness_score + accuracy_score + consistency_score + timeliness_score) / 4
        quality_level = self._determine_quality_level(overall_score)
        
        return DataQualityReport(
            symbol=symbol or "unknown",
            source="unknown",
            timestamp=datetime.now(),
            quality_level=quality_level,
            completeness_score=completeness_score,
            accuracy_score=accuracy_score,
            consistency_score=consistency_score,
            timeliness_score=timeliness_score,
            anomalies_detected=len(anomalies),
            missing_data_points=missing_data,
            data_gaps=data_gaps,
            quality_issues=quality_issues,
            recommendations=recommendations
        )
    
    def _calculate_completeness_score(self, df: pd.DataFrame) -> float:
        """Calculate data completeness score."""
        if df.empty:
            return 0.0
        
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        completeness = (total_cells - missing_cells) / total_cells
        
        return min(completeness, 1.0)
    
    def _calculate_accuracy_score(self, df: pd.DataFrame) -> float:
        """Calculate data accuracy score."""
        if df.empty:
            return 0.0
        
        accuracy_issues = 0
        total_checks = 0
        
        # Check for negative prices
        if 'Close' in df.columns:
            total_checks += len(df)
            accuracy_issues += (df['Close'] <= 0).sum()
        
        # Check OHLC relationships
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            total_checks += len(df) * 2
            accuracy_issues += ((df['Low'] > df['Open']) | (df['Low'] > df['Close'])).sum()
            accuracy_issues += ((df['High'] < df['Open']) | (df['High'] < df['Close'])).sum()
        
        # Check volume
        if 'Volume' in df.columns:
            total_checks += len(df)
            accuracy_issues += (df['Volume'] < 0).sum()
        
        if total_checks == 0:
            return 1.0
        
        accuracy = (total_checks - accuracy_issues) / total_checks
        return min(accuracy, 1.0)
    
    def _calculate_consistency_score(self, df: pd.DataFrame) -> float:
        """Calculate data consistency score."""
        if df.empty or len(df) < 2:
            return 1.0
        
        consistency_issues = 0
        total_checks = 0
        
        # Check for extreme price changes
        if 'Close' in df.columns:
            price_changes = df['Close'].pct_change().abs()
            total_checks += len(price_changes.dropna())
            consistency_issues += (price_changes > 0.5).sum()  # 50% change threshold
        
        # Check for volume consistency
        if 'Volume' in df.columns:
            volume_changes = df['Volume'].pct_change().abs()
            total_checks += len(volume_changes.dropna())
            consistency_issues += (volume_changes > 10.0).sum()  # 1000% volume change threshold
        
        if total_checks == 0:
            return 1.0
        
        consistency = (total_checks - consistency_issues) / total_checks
        return min(consistency, 1.0)
    
    def _calculate_timeliness_score(self, df: pd.DataFrame) -> float:
        """Calculate data timeliness score."""
        if df.empty:
            return 0.0
        
        # Check for recent data
        if df.index.dtype == 'datetime64[ns]':
            latest_date = df.index.max()
            days_old = (datetime.now() - latest_date).days
            
            if days_old <= 1:
                return 1.0
            elif days_old <= 7:
                return 0.8
            elif days_old <= 30:
                return 0.6
            else:
                return 0.2
        
        return 0.5  # Default score if no datetime index
    
    def _detect_anomalies(self, df: pd.DataFrame) -> List[int]:
        """Detect anomalies in price data using multiple methods."""
        if df.empty or len(df) < 10:
            return []
        
        anomalies = set()
        
        # Method 1: Statistical outliers (Z-score)
        if 'Close' in df.columns:
            z_scores = np.abs(stats.zscore(df['Close'].dropna()))
            outliers = np.where(z_scores > 3)[0]
            anomalies.update(outliers)
        
        # Method 2: Isolation Forest
        try:
            if len(df) > 20 and all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                features = df[['Open', 'High', 'Low', 'Close', 'Volume']].fillna(0)
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)
                
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                anomaly_labels = iso_forest.fit_predict(features_scaled)
                iso_outliers = np.where(anomaly_labels == -1)[0]
                anomalies.update(iso_outliers)
        except Exception as e:
            logger.warning(f"Isolation Forest anomaly detection failed: {e}")
        
        # Method 3: Price change anomalies
        if 'Close' in df.columns and len(df) > 1:
            price_changes = df['Close'].pct_change().abs()
            change_threshold = price_changes.quantile(0.99)  # 99th percentile
            change_outliers = np.where(price_changes > change_threshold)[0]
            anomalies.update(change_outliers)
        
        return list(anomalies)
    
    def _find_missing_data(self, df: pd.DataFrame) -> int:
        """Find number of missing data points."""
        return df.isnull().sum().sum()
    
    def _find_data_gaps(self, df: pd.DataFrame) -> List[Tuple[datetime, datetime]]:
        """Find gaps in time series data."""
        if df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return []
        
        gaps = []
        sorted_index = df.index.sort_values()
        
        for i in range(len(sorted_index) - 1):
            current_date = sorted_index[i]
            next_date = sorted_index[i + 1]
            
            # Check for gaps larger than expected interval
            expected_interval = pd.Timedelta(days=1)  # Daily data
            actual_interval = next_date - current_date
            
            if actual_interval > expected_interval * 2:  # More than 2 days gap
                gaps.append((current_date, next_date))
        
        return gaps
    
    def _identify_quality_issues(self, df: pd.DataFrame, completeness: float, 
                               accuracy: float, consistency: float, timeliness: float) -> List[str]:
        """Identify specific quality issues."""
        issues = []
        
        if completeness < self.quality_thresholds['completeness']:
            issues.append(f"Low completeness: {completeness:.2%} (threshold: {self.quality_thresholds['completeness']:.2%})")
        
        if accuracy < self.quality_thresholds['accuracy']:
            issues.append(f"Low accuracy: {accuracy:.2%} (threshold: {self.quality_thresholds['accuracy']:.2%})")
        
        if consistency < self.quality_thresholds['consistency']:
            issues.append(f"Low consistency: {consistency:.2%} (threshold: {self.quality_thresholds['consistency']:.2%})")
        
        if timeliness < self.quality_thresholds['timeliness']:
            issues.append(f"Low timeliness: {timeliness:.2%} (threshold: {self.quality_thresholds['timeliness']:.2%})")
        
        # Check for specific data issues
        if 'Close' in df.columns:
            if (df['Close'] <= 0).any():
                issues.append("Negative or zero prices detected")
            
            if df['Close'].isnull().any():
                issues.append("Missing price data")
        
        if 'Volume' in df.columns:
            if (df['Volume'] < 0).any():
                issues.append("Negative volume detected")
        
        return issues
    
    def _generate_recommendations(self, issues: List[str], anomalies: List[int], 
                                missing_data: int) -> List[str]:
        """Generate recommendations based on quality issues."""
        recommendations = []
        
        if "Low completeness" in str(issues):
            recommendations.append("Implement data backfill from alternative sources")
        
        if "Low accuracy" in str(issues):
            recommendations.append("Review data source configuration and validation rules")
        
        if "Low consistency" in str(issues):
            recommendations.append("Implement data smoothing and outlier detection")
        
        if "Low timeliness" in str(issues):
            recommendations.append("Improve data update frequency and real-time feeds")
        
        if len(anomalies) > 0:
            recommendations.append(f"Review {len(anomalies)} detected anomalies for data quality")
        
        if missing_data > 0:
            recommendations.append(f"Address {missing_data} missing data points")
        
        if not recommendations:
            recommendations.append("Data quality is acceptable")
        
        return recommendations
    
    def _determine_quality_level(self, overall_score: float) -> DataQualityLevel:
        """Determine overall quality level based on score."""
        if overall_score >= 0.95:
            return DataQualityLevel.EXCELLENT
        elif overall_score >= 0.85:
            return DataQualityLevel.GOOD
        elif overall_score >= 0.70:
            return DataQualityLevel.FAIR
        elif overall_score >= 0.50:
            return DataQualityLevel.POOR
        else:
            return DataQualityLevel.CRITICAL
    
    def _create_empty_report(self, symbol: str, source: str) -> DataQualityReport:
        """Create empty quality report for failed validation."""
        return DataQualityReport(
            symbol=symbol,
            source=source,
            timestamp=datetime.now(),
            quality_level=DataQualityLevel.CRITICAL,
            completeness_score=0.0,
            accuracy_score=0.0,
            consistency_score=0.0,
            timeliness_score=0.0,
            anomalies_detected=0,
            missing_data_points=0,
            data_gaps=[],
            quality_issues=["No data available"],
            recommendations=["Implement data source or check connectivity"]
        )


class DataEnhancer:
    """Data enhancement and cleaning utilities."""
    
    def __init__(self):
        self.validator = DataQualityValidator()
    
    def clean_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and enhance price data."""
        if df.empty:
            return df
        
        cleaned_df = df.copy()
        
        # Remove negative prices
        if 'Close' in cleaned_df.columns:
            cleaned_df = cleaned_df[cleaned_df['Close'] > 0]
        
        # Fix OHLC relationships
        if all(col in cleaned_df.columns for col in ['Open', 'High', 'Low', 'Close']):
            # Ensure Low <= min(Open, Close) and High >= max(Open, Close)
            cleaned_df['Low'] = cleaned_df[['Open', 'Close', 'Low']].min(axis=1)
            cleaned_df['High'] = cleaned_df[['Open', 'Close', 'High']].max(axis=1)
        
        # Remove negative volume
        if 'Volume' in cleaned_df.columns:
            cleaned_df.loc[cleaned_df['Volume'] < 0, 'Volume'] = 0
        
        # Fill missing values with forward fill
        cleaned_df = cleaned_df.fillna(method='ffill')
        
        # Remove remaining NaN values
        cleaned_df = cleaned_df.dropna()
        
        return cleaned_df
    
    def interpolate_missing_data(self, df: pd.DataFrame, method: str = 'linear') -> pd.DataFrame:
        """Interpolate missing data points."""
        if df.empty:
            return df
        
        # Only interpolate numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df_interpolated = df.copy()
        
        for col in numeric_columns:
            if df_interpolated[col].isnull().any():
                df_interpolated[col] = df_interpolated[col].interpolate(method=method)
        
        return df_interpolated
    
    def remove_outliers(self, df: pd.DataFrame, columns: List[str] = None, 
                       method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """Remove outliers from data."""
        if df.empty:
            return df
        
        if columns is None:
            columns = ['Close'] if 'Close' in df.columns else df.select_dtypes(include=[np.number]).columns.tolist()
        
        cleaned_df = df.copy()
        
        for col in columns:
            if col not in cleaned_df.columns:
                continue
            
            if method == 'iqr':
                Q1 = cleaned_df[col].quantile(0.25)
                Q3 = cleaned_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
            
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(cleaned_df[col].dropna()))
                cleaned_df = cleaned_df[z_scores < threshold]
        
        return cleaned_df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators to price data."""
        if df.empty or 'Close' not in df.columns:
            return df
        
        enhanced_df = df.copy()
        
        # Simple Moving Averages
        enhanced_df['SMA_20'] = enhanced_df['Close'].rolling(window=20).mean()
        enhanced_df['SMA_50'] = enhanced_df['Close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        enhanced_df['EMA_12'] = enhanced_df['Close'].ewm(span=12).mean()
        enhanced_df['EMA_26'] = enhanced_df['Close'].ewm(span=26).mean()
        
        # RSI
        delta = enhanced_df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        enhanced_df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        enhanced_df['BB_Middle'] = enhanced_df['Close'].rolling(window=20).mean()
        bb_std = enhanced_df['Close'].rolling(window=20).std()
        enhanced_df['BB_Upper'] = enhanced_df['BB_Middle'] + (bb_std * 2)
        enhanced_df['BB_Lower'] = enhanced_df['BB_Middle'] - (bb_std * 2)
        
        return enhanced_df


class DataLineageTracker:
    """Track data lineage and provenance."""
    
    def __init__(self):
        self.lineage_records = []
    
    def record_data_operation(self, operation: str, input_sources: List[str], 
                            output_data: str, timestamp: datetime = None):
        """Record a data operation for lineage tracking."""
        if timestamp is None:
            timestamp = datetime.now()
        
        record = {
            'timestamp': timestamp,
            'operation': operation,
            'input_sources': input_sources,
            'output_data': output_data
        }
        
        self.lineage_records.append(record)
        logger.info(f"Recorded data operation: {operation}")
    
    def get_lineage_report(self) -> List[Dict[str, Any]]:
        """Get complete lineage report."""
        return self.lineage_records.copy()
    
    def export_lineage(self, filepath: str):
        """Export lineage data to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.lineage_records, f, indent=2, default=str)
        logger.info(f"Exported lineage data to {filepath}")


def create_data_quality_system() -> Tuple[DataQualityValidator, DataEnhancer, DataLineageTracker]:
    """Create complete data quality system."""
    validator = DataQualityValidator()
    enhancer = DataEnhancer()
    tracker = DataLineageTracker()
    
    logger.info("Data quality system created successfully")
    return validator, enhancer, tracker
