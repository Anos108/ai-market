import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SystemStatusService, AnalyticsResponse, AgentPerformanceMetrics, MarketTrends, SystemAnalytics } from '../../services/system-status.service';
import { LoadingComponent } from '../../shared/loading/loading.component';
import { SkeletonComponent } from '../../shared/skeleton/skeleton.component';
import { Observable } from 'rxjs';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../../environments/environment';

@Component({
  selector: 'app-analytics',
  standalone: true,
  imports: [CommonModule, LoadingComponent, SkeletonComponent],
  template: `
    <div class="space-y-6 max-w-7xl mx-auto">
      <!-- Page Header -->
      <div>
        <h1 class="text-3xl font-bold text-gray-900">Analytics</h1>
        <p class="text-gray-600">Advanced analytics and insights</p>
      </div>

      <!-- Analytics Overview Cards -->
      <div *ngIf="!isLoadingAnalytics; else analyticsLoading" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <ng-container *ngIf="analytics$ | async as analytics">
        <div class="metric-card-enhanced">
          <div class="flex items-center justify-between">
            <div class="flex items-center">
              <div class="flex-shrink-0">
                <div class="w-12 h-12 bg-gradient-to-br from-blue-400 to-blue-600 rounded-xl flex items-center justify-center shadow-lg">
                  <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
                  </svg>
                </div>
              </div>
              <div class="ml-4">
                <p class="text-sm font-medium text-gray-600">Total Predictions</p>
                <p class="text-2xl font-bold text-gray-900">{{ analytics.total_predictions | number }}</p>
              </div>
            </div>
            <div class="flex items-center">
              <div class="w-3 h-3 bg-blue-500 rounded-full animate-pulse"></div>
            </div>
          </div>
        </div>

        <div class="metric-card-enhanced">
          <div class="flex items-center justify-between">
            <div class="flex items-center">
              <div class="flex-shrink-0">
                <div class="w-12 h-12 bg-gradient-to-br from-green-400 to-green-600 rounded-xl flex items-center justify-center shadow-lg">
                  <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path>
                  </svg>
                </div>
              </div>
              <div class="ml-4">
                <p class="text-sm font-medium text-gray-600">Avg Accuracy</p>
                <p class="text-2xl font-bold text-gray-900">{{ analytics.accuracy_rate | percent:'1.1-1' }}</p>
              </div>
            </div>
            <div class="text-right">
              <p class="text-xs text-gray-500">Market Trend</p>
              <p class="text-sm font-semibold" [class]="getTrendClass(analytics.market_trends.trend_direction)">
                {{ analytics.market_trends.trend_direction }}
              </p>
            </div>
          </div>
        </div>

        <div class="metric-card-enhanced">
          <div class="flex items-center justify-between">
            <div class="flex items-center">
              <div class="flex-shrink-0">
                <div class="w-12 h-12 bg-gradient-to-br from-purple-400 to-purple-600 rounded-xl flex items-center justify-center shadow-lg">
                  <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                  </svg>
                </div>
              </div>
              <div class="ml-4">
                <p class="text-sm font-medium text-gray-600">Response Time</p>
                <p class="text-2xl font-bold text-gray-900">{{ analytics.system_analytics.avg_response_time }}ms</p>
              </div>
            </div>
            <div class="text-right">
              <p class="text-xs text-gray-500">System Load</p>
              <p class="text-sm font-semibold text-purple-600">{{ (analytics.system_analytics.system_load * 100) | number:'1.0-0' }}%</p>
            </div>
          </div>
        </div>

        <div class="metric-card-enhanced">
          <div class="flex items-center justify-between">
            <div class="flex items-center">
              <div class="flex-shrink-0">
                <div class="w-12 h-12 bg-gradient-to-br from-amber-400 to-amber-600 rounded-xl flex items-center justify-center shadow-lg">
                  <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"></path>
                  </svg>
                </div>
              </div>
              <div class="ml-4">
                <p class="text-sm font-medium text-gray-600">System Reliability</p>
                <p class="text-2xl font-bold text-gray-900">{{ (analytics.system_analytics.system_reliability * 100) | number:'1.1-1' }}%</p>
              </div>
            </div>
            <div class="text-right">
              <p class="text-xs text-gray-500">Uptime</p>
              <p class="text-sm font-semibold text-amber-600">{{ analytics.system_analytics.total_uptime_hours | number:'1.1-1' }}h</p>
            </div>
          </div>
        </div>
        </ng-container>
      </div>

      <ng-template #analyticsLoading>
        <app-skeleton type="metrics" [metrics]="[1,2,3,4]"></app-skeleton>
      </ng-template>

      <!-- Analytics Content -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <!-- Performance Metrics -->
        <div class="card-enhanced">
          <div class="card-header-enhanced">
            <div class="flex items-center justify-between">
              <div>
                <h3 class="text-xl font-bold text-gray-900">Performance Metrics</h3>
                <p class="text-sm text-gray-600 mt-1">Key performance indicators</p>
              </div>
              <div class="flex items-center space-x-2">
                <div class="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span class="text-sm font-medium text-gray-600">Live</span>
              </div>
            </div>
          </div>
          <div class="p-6">
            <div class="space-y-4">
              <div class="flex justify-between items-center">
                <span class="text-sm text-gray-600">Model Accuracy</span>
                <span class="font-semibold text-green-600">87.3%</span>
              </div>
              <div class="flex justify-between items-center">
                <span class="text-sm text-gray-600">Prediction Speed</span>
                <span class="font-semibold text-blue-600">1.2s avg</span>
              </div>
              <div class="flex justify-between items-center">
                <span class="text-sm text-gray-600">Data Coverage</span>
                <span class="font-semibold text-purple-600">94.7%</span>
              </div>
              <div class="flex justify-between items-center">
                <span class="text-sm text-gray-600">Uptime</span>
                <span class="font-semibold text-green-600">99.9%</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Data Sources -->
        <div class="card-enhanced">
          <div class="card-header-enhanced">
            <div class="flex items-center justify-between">
              <div>
                <h3 class="text-xl font-bold text-gray-900">Data Sources</h3>
                <p class="text-sm text-gray-600 mt-1">Connected data feeds</p>
              </div>
              <div class="flex items-center space-x-2">
                <div class="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                <span class="text-sm font-medium text-gray-600">{{ dataSources.length }} Active</span>
              </div>
            </div>
          </div>
          <div class="p-6">
            <div *ngIf="loadingDataSources" class="flex justify-center items-center py-8">
              <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
              <span class="ml-2 text-gray-600">Loading data sources...</span>
            </div>
            <div *ngIf="!loadingDataSources" class="space-y-4 max-h-96 overflow-y-auto">
              <div *ngFor="let source of dataSources" class="flex justify-between items-center">
                <div class="flex items-center">
                  <div class="w-8 h-8 bg-gradient-to-br rounded-lg flex items-center justify-center mr-3" [class]="getSourceColor(source.name)">
                    <span class="text-white text-xs font-bold">{{ getSourceIcon(source.name) }}</span>
                  </div>
                  <div>
                    <span class="font-semibold text-gray-900">{{ source.name }}</span>
                    <p class="text-xs text-gray-500">{{ source.type }}</p>
                  </div>
                </div>
                <div class="text-right">
                  <span class="text-sm font-semibold" [class]="source.status === 'active' ? 'text-green-600' : 'text-red-600'">
                    {{ source.status | titlecase }}
                  </span>
                  <p class="text-xs text-gray-500">{{ source.last_update | date:'short' }}</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Data Quality -->
        <div class="card-enhanced">
          <div class="card-header-enhanced">
            <div class="flex items-center justify-between">
              <div>
                <h3 class="text-xl font-bold text-gray-900">Data Quality</h3>
                <p class="text-sm text-gray-600 mt-1">Quality metrics and validation</p>
              </div>
              <div class="flex items-center space-x-2">
                <div class="w-2 h-2 rounded-full animate-pulse" [class]="getQualityStatusColor(dataQuality?.quality_level)"></div>
                <span class="text-sm font-medium text-gray-600">{{ dataQuality?.quality_level | titlecase }}</span>
              </div>
            </div>
          </div>
          <div class="p-6">
            <div *ngIf="loadingDataQuality" class="flex justify-center items-center py-8">
              <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
              <span class="ml-2 text-gray-600">Loading data quality...</span>
            </div>
            <div *ngIf="!loadingDataQuality && dataQuality" class="space-y-4">
              <div class="grid grid-cols-2 gap-4">
                <div class="text-center">
                  <p class="text-2xl font-bold text-blue-600">{{ (dataQuality.overall_score * 100) | number:'1.1-1' }}%</p>
                  <p class="text-sm text-gray-600">Overall Score</p>
                </div>
                <div class="text-center">
                  <p class="text-2xl font-bold text-green-600">{{ (dataQuality.completeness * 100) | number:'1.1-1' }}%</p>
                  <p class="text-sm text-gray-600">Completeness</p>
                </div>
              </div>
              <div class="grid grid-cols-2 gap-4">
                <div class="text-center">
                  <p class="text-2xl font-bold text-purple-600">{{ (dataQuality.accuracy * 100) | number:'1.1-1' }}%</p>
                  <p class="text-sm text-gray-600">Accuracy</p>
                </div>
                <div class="text-center">
                  <p class="text-2xl font-bold text-amber-600">{{ (dataQuality.consistency * 100) | number:'1.1-1' }}%</p>
                  <p class="text-sm text-gray-600">Consistency</p>
                </div>
              </div>
              <div class="pt-4 border-t border-gray-200">
                <div class="flex justify-between items-center mb-2">
                  <span class="text-sm text-gray-600">Anomalies Detected</span>
                  <span class="text-sm font-semibold text-orange-600">{{ dataQuality.anomalies_detected }}</span>
                </div>
                <div class="flex justify-between items-center mb-2">
                  <span class="text-sm text-gray-600">Missing Data Points</span>
                  <span class="text-sm font-semibold text-red-600">{{ dataQuality.missing_data_points }}</span>
                </div>
                <div class="flex justify-between items-center">
                  <span class="text-sm text-gray-600">Data Gaps</span>
                  <span class="text-sm font-semibold text-yellow-600">{{ dataQuality.data_gaps }}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `,
  styles: []
})
export class AnalyticsComponent implements OnInit {
  analytics$: Observable<AnalyticsResponse> | undefined;
  agentPerformance$: Observable<AgentPerformanceMetrics[]> | undefined;
  marketTrends$: Observable<MarketTrends> | undefined;
  systemMetrics$: Observable<SystemAnalytics> | undefined;
  dataSources: any[] = [];
  dataQuality: any = null;
  loadingDataSources = false;
  loadingDataQuality = false;
  isLoadingAnalytics = true;

  constructor(private systemStatusService: SystemStatusService, private http: HttpClient) {}

  ngOnInit() {
    this.analytics$ = this.systemStatusService.getAnalytics();
    this.analytics$.subscribe({
      next: () => {
        this.isLoadingAnalytics = false;
      },
      error: () => {
        this.isLoadingAnalytics = false;
      }
    });
    this.agentPerformance$ = this.systemStatusService.getAgentPerformance();
    this.marketTrends$ = this.systemStatusService.getMarketTrends();
    this.systemMetrics$ = this.systemStatusService.getSystemMetrics();
    this.loadDataSources();
    this.loadDataQuality();
  }

  getTotalPredictions(agentPerformance: AgentPerformanceMetrics[]): number {
    return agentPerformance.reduce((total, agent) => total + agent.total_predictions, 0);
  }

  getAverageAccuracy(agentPerformance: AgentPerformanceMetrics[]): number {
    if (agentPerformance.length === 0) return 0;
    const totalAccuracy = agentPerformance.reduce((sum, agent) => sum + agent.accuracy, 0);
    return totalAccuracy / agentPerformance.length;
  }

  getTrendClass(trendDirection: string): string {
    switch (trendDirection.toLowerCase()) {
      case 'bullish':
        return 'text-green-600';
      case 'bearish':
        return 'text-red-600';
      default:
        return 'text-gray-600';
    }
  }

  formatNumber(value: number): string {
    return new Intl.NumberFormat('en-US').format(value);
  }

  formatPercent(value: number): string {
    return `${(value * 100).toFixed(1)}%`;
  }

  loadDataSources() {
    this.loadingDataSources = true;
    this.http.get<any>(`${environment.apiUrl}/data-sources`).subscribe({
      next: (response) => {
        this.dataSources = response.sources || [];
        this.loadingDataSources = false;
      },
      error: (error) => {
        console.error('Error loading data sources:', error);
        this.loadingDataSources = false;
      }
    });
  }

  loadDataQuality() {
    this.loadingDataQuality = true;
    this.http.get<any>(`${environment.apiUrl}/data-quality`).subscribe({
      next: (response) => {
        this.dataQuality = response;
        this.loadingDataQuality = false;
      },
      error: (error) => {
        console.error('Error loading data quality:', error);
        this.loadingDataQuality = false;
      }
    });
  }

  getSourceIcon(sourceName: string): string {
    const icons: { [key: string]: string } = {
      'Yahoo Finance': 'YF',
      'Alpha Vantage': 'AV',
      'IEX Cloud': 'IEX',
      'Polygon.io': 'POL',
      'FRED': 'FRED',
      'Binance': 'BIN',
      'Coinbase': 'CB',
      'CoinGecko': 'CG',
      'News API': 'NEWS',
      'Economic Data': 'ECON',
      'Social Sentiment': 'SOC',
      'News Sentiment': 'NS',
      'Satellite Data': 'SAT',
      'Credit Card Data': 'CC',
      'Supply Chain': 'SC'
    };
    return icons[sourceName] || sourceName.substring(0, 2).toUpperCase();
  }

  getSourceColor(sourceName: string): string {
    const colors: { [key: string]: string } = {
      'Yahoo Finance': 'from-green-400 to-green-600',
      'Alpha Vantage': 'from-blue-400 to-blue-600',
      'IEX Cloud': 'from-purple-400 to-purple-600',
      'Polygon.io': 'from-indigo-400 to-indigo-600',
      'FRED': 'from-red-400 to-red-600',
      'Binance': 'from-yellow-400 to-yellow-600',
      'Coinbase': 'from-blue-400 to-blue-600',
      'CoinGecko': 'from-orange-400 to-orange-600',
      'News API': 'from-pink-400 to-pink-600',
      'Economic Data': 'from-gray-400 to-gray-600',
      'Social Sentiment': 'from-cyan-400 to-cyan-600',
      'News Sentiment': 'from-teal-400 to-teal-600',
      'Satellite Data': 'from-emerald-400 to-emerald-600',
      'Credit Card Data': 'from-violet-400 to-violet-600',
      'Supply Chain': 'from-amber-400 to-amber-600'
    };
    return colors[sourceName] || 'from-gray-400 to-gray-600';
  }

  getQualityStatusColor(qualityLevel: string): string {
    switch (qualityLevel?.toLowerCase()) {
      case 'excellent': return 'bg-green-500';
      case 'good': return 'bg-blue-500';
      case 'fair': return 'bg-yellow-500';
      case 'poor': return 'bg-orange-500';
      case 'critical': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  }
}