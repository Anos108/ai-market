import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { interval, Subscription } from 'rxjs';
import { switchMap, catchError } from 'rxjs/operators';
import { of } from 'rxjs';

import { 
  EnsembleBlenderSummary, 
  EnsembleSignalData, 
  SignalQualityData, 
  EnsemblePerformanceData 
} from '../../interfaces/ensemble-blender.interface';

@Component({
  selector: 'app-ensemble-blender',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './ensemble-blender.component.html',
  styleUrls: ['./ensemble-blender.component.css']
})
export class EnsembleBlenderComponent implements OnInit, OnDestroy {
  // Data properties
  summary: EnsembleBlenderSummary | null = null;
  signals: EnsembleSignalData[] = [];
  qualityMetrics: SignalQualityData | null = null;
  performanceData: EnsemblePerformanceData | null = null;
  
  // Loading states
  loadingSummary = true;
  loadingSignals = true;
  loadingQuality = true;
  loadingPerformance = true;
  
  // Error states
  errorSummary = false;
  errorSignals = false;
  errorQuality = false;
  errorPerformance = false;
  
  // Auto-refresh
  private refreshSubscription: Subscription | null = null;
  private readonly REFRESH_INTERVAL = 30000; // 30 seconds
  
  constructor(private http: HttpClient) {}
  
  ngOnInit(): void {
    this.loadAllData();
    this.startAutoRefresh();
  }
  
  ngOnDestroy(): void {
    if (this.refreshSubscription) {
      this.refreshSubscription.unsubscribe();
    }
  }
  
  private loadAllData(): void {
    this.loadSummary();
    this.loadSignals();
    this.loadQualityMetrics();
    this.loadPerformanceData();
  }
  
  private startAutoRefresh(): void {
    this.refreshSubscription = interval(this.REFRESH_INTERVAL)
      .subscribe(() => {
        this.loadAllData();
      });
  }
  
  private loadSummary(): void {
    this.loadingSummary = true;
    this.errorSummary = false;
    
    this.http.get<EnsembleBlenderSummary>('http://localhost:8001/ensemble-blender')
      .pipe(
        catchError(error => {
          console.error('Error loading ensemble blender summary:', error);
          this.errorSummary = true;
          return of(null);
        })
      )
      .subscribe(data => {
        this.summary = data;
        this.loadingSummary = false;
      });
  }
  
  private loadSignals(): void {
    this.loadingSignals = true;
    this.errorSignals = false;
    
    this.http.get<EnsembleSignalData[]>('http://localhost:8001/ensemble-blender/signals')
      .pipe(
        catchError(error => {
          console.error('Error loading ensemble signals:', error);
          this.errorSignals = true;
          return of([]);
        })
      )
      .subscribe(data => {
        this.signals = data;
        this.loadingSignals = false;
      });
  }
  
  private loadQualityMetrics(): void {
    this.loadingQuality = true;
    this.errorQuality = false;
    
    this.http.get<SignalQualityData>('http://localhost:8001/ensemble-blender/quality')
      .pipe(
        catchError(error => {
          console.error('Error loading quality metrics:', error);
          this.errorQuality = true;
          return of(null);
        })
      )
      .subscribe(data => {
        this.qualityMetrics = data;
        this.loadingQuality = false;
      });
  }
  
  private loadPerformanceData(): void {
    this.loadingPerformance = true;
    this.errorPerformance = false;
    
    this.http.get<EnsemblePerformanceData>('http://localhost:8001/ensemble-blender/performance')
      .pipe(
        catchError(error => {
          console.error('Error loading performance data:', error);
          this.errorPerformance = true;
          return of(null);
        })
      )
      .subscribe(data => {
        this.performanceData = data;
        this.loadingPerformance = false;
      });
  }
  
  refreshData(): void {
    this.loadAllData();
  }
  
  getSignalTypeColor(signalType: string): string {
    switch (signalType.toLowerCase()) {
      case 'strong_buy':
        return 'text-green-600 bg-green-100';
      case 'buy':
        return 'text-green-500 bg-green-50';
      case 'hold':
        return 'text-gray-600 bg-gray-100';
      case 'sell':
        return 'text-red-500 bg-red-50';
      case 'strong_sell':
        return 'text-red-600 bg-red-100';
      default:
        return 'text-gray-500 bg-gray-50';
    }
  }
  
  getRegimeColor(regime: string): string {
    switch (regime.toLowerCase()) {
      case 'bull':
        return 'text-green-600 bg-green-100';
      case 'bear':
        return 'text-red-600 bg-red-100';
      case 'sideways':
        return 'text-gray-600 bg-gray-100';
      case 'volatile':
        return 'text-yellow-600 bg-yellow-100';
      case 'trending':
        return 'text-blue-600 bg-blue-100';
      default:
        return 'text-gray-500 bg-gray-50';
    }
  }
  
  getBlendModeColor(blendMode: string): string {
    switch (blendMode.toLowerCase()) {
      case 'weighted_average':
        return 'text-blue-600 bg-blue-100';
      case 'majority':
        return 'text-purple-600 bg-purple-100';
      case 'max_confidence':
        return 'text-orange-600 bg-orange-100';
      case 'average':
        return 'text-green-600 bg-green-100';
      default:
        return 'text-gray-500 bg-gray-50';
    }
  }
  
  getQualityScoreColor(score: number): string {
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  }
  
  formatTimestamp(timestamp: string): string {
    return new Date(timestamp).toLocaleString();
  }
  
  formatPercentage(value: number): string {
    if (value == null || isNaN(value)) {
      return 'N/A';
    }
    return (value * 100).toFixed(1) + '%';
  }
}
