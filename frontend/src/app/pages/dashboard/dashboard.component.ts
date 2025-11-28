import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SystemStatusService, SystemStatus, Prediction } from '../../services/system-status.service';
import { SkeletonComponent } from '../../shared/skeleton/skeleton.component';
import { Subscription } from 'rxjs';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [CommonModule, SkeletonComponent],
  template: `
    <div class="space-y-6 max-w-7xl mx-auto">
      <!-- Page Header -->
      <div class="flex items-center justify-between">
        <div>
          <h1 class="text-3xl font-bold text-gray-900">Dashboard</h1>
          <p class="text-gray-600">AI Market Analysis System Overview</p>
        </div>
        <div class="flex items-center space-x-4">
          <div class="text-sm text-gray-500">
            Last updated: {{ lastUpdated | date:'medium' }}
          </div>
        </div>
      </div>

      <!-- System Status Cards -->
      <div *ngIf="!isLoadingSystemStatus; else systemStatusLoading" class="metric-grid">
        <div class="metric-card">
          <div class="flex items-center">
            <div class="flex-shrink-0">
              <div class="w-8 h-8 bg-primary-100 rounded-lg flex items-center justify-center">
                <svg class="w-5 h-5 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"></path>
                </svg>
              </div>
            </div>
            <div class="ml-4 flex-1 min-w-0">
              <p class="text-sm font-medium text-gray-500 truncate">Active Agents</p>
              <p class="text-2xl font-semibold text-gray-900 truncate">{{ systemStatus?.active_agents?.length || 0 }}</p>
            </div>
          </div>
        </div>

        <div class="metric-card">
          <div class="flex items-center">
            <div class="flex-shrink-0">
              <div class="w-8 h-8 bg-success-100 rounded-lg flex items-center justify-center">
                <svg class="w-5 h-5 text-success-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path>
                </svg>
              </div>
            </div>
            <div class="ml-4 flex-1 min-w-0">
              <p class="text-sm font-medium text-gray-500 truncate">Total Predictions</p>
              <p class="text-2xl font-semibold text-gray-900 truncate">{{ systemStatus?.total_predictions || 0 }}</p>
            </div>
          </div>
        </div>

        <div class="metric-card">
          <div class="flex items-center">
            <div class="flex-shrink-0">
              <div class="w-8 h-8 bg-warning-100 rounded-lg flex items-center justify-center">
                <svg class="w-5 h-5 text-warning-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                </svg>
              </div>
            </div>
            <div class="ml-4 flex-1 min-w-0">
              <p class="text-sm font-medium text-gray-500 truncate">System Uptime</p>
              <p class="text-2xl font-semibold text-gray-900 truncate">{{ formatUptime(systemStatus?.uptime_seconds) }}</p>
            </div>
          </div>
        </div>

        <div class="metric-card">
          <div class="flex items-center">
            <div class="flex-shrink-0">
              <div class="w-8 h-8 bg-info-100 rounded-lg flex items-center justify-center">
                <svg class="w-5 h-5 text-info-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
                </svg>
              </div>
            </div>
            <div class="ml-4 flex-1 min-w-0">
              <p class="text-sm font-medium text-gray-500 truncate">Data Quality</p>
              <p class="text-2xl font-semibold text-gray-900 truncate">{{ formatPercentage(systemStatus?.data_quality_score) }}%</p>
            </div>
          </div>
        </div>
      </div>

      <ng-template #systemStatusLoading>
        <app-skeleton type="metrics" [metrics]="[1,2,3,4]"></app-skeleton>
      </ng-template>


      <!-- Top Holdings -->
      <div class="card">
        <div class="card-header">
          <h3 class="text-lg font-semibold text-gray-900">Top Holdings</h3>
          <div *ngIf="isLoadingHoldings" class="flex items-center text-sm text-blue-600">
            <svg class="animate-spin -ml-1 mr-2 h-4 w-4 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Loading holdings data...
          </div>
        </div>
        <div *ngIf="!isLoadingHoldings; else holdingsLoading" class="overflow-x-auto">
          <table class="table">
            <thead class="table-header">
              <tr>
                <th class="table-header-cell">Symbol</th>
                <th class="table-header-cell">Price</th>
                <th class="table-header-cell">Change</th>
                <th class="table-header-cell">Value</th>
                <th class="table-header-cell">P&L</th>
              </tr>
            </thead>
            <tbody class="table-body">
              <tr *ngFor="let holding of topHoldings" class="table-row">
                <td class="table-cell font-medium">{{ holding.symbol }}</td>
                <td class="table-cell">{{ holding.current_price | currency:'USD':'symbol':'1.2-2' }}</td>
                <td class="table-cell">
                  <span [class]="holding.change_percent >= 0 ? 'text-green-600' : 'text-red-600'">
                    {{ holding.change_percent >= 0 ? '+' : '' }}{{ holding.change_percent | number:'1.2-2' }}%
                  </span>
                </td>
                <td class="table-cell">{{ holding.market_value | currency:'USD':'symbol':'1.2-2' }}</td>
                <td class="table-cell">
                  <span [class]="holding.pnl >= 0 ? 'text-green-600' : 'text-red-600'">
                    {{ holding.pnl >= 0 ? '+' : '' }}{{ holding.pnl | currency:'USD':'symbol':'1.2-2' }}
                  </span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <ng-template #holdingsLoading>
          <div class="p-6">
            <div class="flex items-center justify-center py-8">
              <div class="flex items-center space-x-3">
                <svg class="animate-spin h-8 w-8 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                  <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span class="text-lg font-medium text-gray-600">Loading Top Holdings...</span>
              </div>
            </div>
          </div>
        </ng-template>
      </div>

      <!-- Trading Signals Summary -->
      <div class="card">
        <div class="card-header">
          <h3 class="text-lg font-semibold text-gray-900">Trading Signals Summary</h3>
        </div>
        <div *ngIf="!isLoadingSignals; else signalsLoading" class="p-6">
          <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div class="text-center">
              <div class="text-2xl font-bold text-green-600">{{ signalSummary?.buy_signals || 0 }}</div>
              <div class="text-sm text-gray-500">Buy Signals</div>
            </div>
            <div class="text-center">
              <div class="text-2xl font-bold text-red-600">{{ signalSummary?.sell_signals || 0 }}</div>
              <div class="text-sm text-gray-500">Sell Signals</div>
            </div>
            <div class="text-center">
              <div class="text-2xl font-bold text-gray-600">{{ signalSummary?.hold_signals || 0 }}</div>
              <div class="text-sm text-gray-500">Hold Signals</div>
            </div>
            <div class="text-center">
              <div class="text-2xl font-bold text-blue-600">{{ signalSummary?.total_signals || 0 }}</div>
              <div class="text-sm text-gray-500">Total Signals</div>
            </div>
          </div>
        </div>
        <ng-template #signalsLoading>
          <div class="p-6">
            <app-skeleton type="metrics" [metrics]="[1,2,3,4]"></app-skeleton>
          </div>
        </ng-template>
      </div>

      <!-- Managed Symbols Insights -->
      <div class="card">
        <div class="card-header">
          <h3 class="text-lg font-semibold text-gray-900">Managed Symbols Insights</h3>
        </div>
        <div *ngIf="!isLoadingSymbols; else symbolsLoading" class="overflow-x-auto">
          <table class="table">
            <thead class="table-header">
              <tr>
                <th class="table-header-cell">Symbol</th>
                <th class="table-header-cell">Current Price</th>
                <th class="table-header-cell">Change</th>
                <th class="table-header-cell">P&L</th>
                <th class="table-header-cell">Weight</th>
                <th class="table-header-cell">Status</th>
                <th class="table-header-cell">Actions</th>
              </tr>
            </thead>
            <tbody class="table-body">
              <tr *ngFor="let symbol of managedSymbols" class="table-row">
                <td class="table-cell">
                  <div class="flex items-center">
                    <div class="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center mr-3">
                      <span class="text-xs font-bold text-blue-600">{{ symbol.symbol.substring(0, 2) }}</span>
                    </div>
                    <div>
                      <div class="font-medium text-gray-900">{{ symbol.symbol }}</div>
                      <div class="text-sm text-gray-500">{{ symbol.name }}</div>
                    </div>
                  </div>
                </td>
                <td class="table-cell font-medium">
                  <span *ngIf="symbol.current_price > 0">{{ symbol.current_price | currency:'USD':'symbol':'1.2-2' }}</span>
                  <span *ngIf="symbol.current_price === 0" class="text-gray-400">N/A</span>
                </td>
                <td class="table-cell">
                  <span *ngIf="symbol.change_percent !== 0" [class]="symbol.change_percent >= 0 ? 'text-green-600' : 'text-red-600'">
                    {{ symbol.change_percent >= 0 ? '+' : '' }}{{ symbol.change_percent | number:'1.2-2' }}%
                  </span>
                  <span *ngIf="symbol.change_percent === 0" class="text-gray-400">N/A</span>
                </td>
                <td class="table-cell">
                  <span *ngIf="symbol.pnl !== 0" [class]="symbol.pnl >= 0 ? 'text-green-600' : 'text-red-600'">
                    {{ symbol.pnl >= 0 ? '+' : '' }}{{ symbol.pnl | currency:'USD':'symbol':'1.2-2' }}
                  </span>
                  <span *ngIf="symbol.pnl === 0" class="text-gray-400">N/A</span>
                </td>
                <td class="table-cell">
                  <div class="flex items-center" *ngIf="symbol.weight > 0">
                    <div class="w-16 bg-gray-200 rounded-full h-2 mr-2">
                      <div class="bg-blue-500 h-2 rounded-full" [style.width.%]="symbol.weight"></div>
                    </div>
                    <span class="text-sm text-gray-600">{{ symbol.weight | number:'1.1-1' }}%</span>
                  </div>
                  <span *ngIf="symbol.weight === 0" class="text-gray-400">N/A</span>
                </td>
                <td class="table-cell">
                  <span [class]="getStatusClass(symbol.status)">{{ symbol.status }}</span>
                </td>
                <td class="table-cell">
                  <div class="flex space-x-2">
                    <button class="text-blue-600 hover:text-blue-800 text-sm font-medium">View</button>
                    <button class="text-gray-600 hover:text-gray-800 text-sm font-medium">Edit</button>
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <ng-template #symbolsLoading>
          <div class="p-6">
            <app-skeleton type="table" [tableHeaders]="['Symbol', 'Price', 'Change', 'P&L', 'Weight', 'Status', 'Actions']" [tableRows]="[['', '', '', '', '', '', ''], ['', '', '', '', '', '', ''], ['', '', '', '', '', '', '']]"></app-skeleton>
          </div>
        </ng-template>
      </div>

      <!-- Market Overview -->
      <div class="card">
        <div class="card-header">
          <h3 class="text-lg font-semibold text-gray-900">Market Overview</h3>
        </div>
        <div *ngIf="!isLoadingMarketOverview; else marketOverviewLoading" class="p-6">
          <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 class="text-sm font-medium text-gray-700 mb-3">Market Indices</h3>
              <div class="space-y-2">
                <div *ngFor="let index of marketIndices" class="flex justify-between items-center">
                  <span class="text-sm text-gray-600">{{ index.name }}</span>
                  <div class="flex items-center space-x-2">
                    <span class="text-sm font-medium">{{ index.value | number:'1.2-2' }}</span>
                    <span class="text-xs" [class]="index.change >= 0 ? 'text-green-600' : 'text-red-600'">
                      {{ index.change >= 0 ? '+' : '' }}{{ index.change | number:'1.2-2' }}%
                    </span>
                  </div>
                </div>
              </div>
            </div>
            <div>
              <h3 class="text-sm font-medium text-gray-700 mb-3">Sector Performance</h3>
              <div class="space-y-2">
                <div *ngFor="let sector of sectorPerformance" class="flex justify-between items-center">
                  <span class="text-sm text-gray-600">{{ sector.name }}</span>
                  <div class="flex items-center space-x-2">
                    <span class="text-sm font-medium">{{ sector.performance | number:'1.2-2' }}%</span>
                    <div class="w-16 bg-gray-200 rounded-full h-1">
                      <div class="h-1 rounded-full" [class]="sector.performance >= 0 ? 'bg-green-500' : 'bg-red-500'" 
                           [style.width.%]="Math.abs(sector.performance) * 2"></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        <ng-template #marketOverviewLoading>
          <div class="p-6">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <div class="skeleton skeleton-title" style="width: 40%; margin-bottom: 1rem;"></div>
                <div class="space-y-2">
                  <div *ngFor="let i of [1,2,3,4]" class="flex justify-between items-center">
                    <div class="skeleton skeleton-text" style="width: 30%;"></div>
                    <div class="skeleton skeleton-text" style="width: 40%;"></div>
                  </div>
                </div>
              </div>
              <div>
                <div class="skeleton skeleton-title" style="width: 50%; margin-bottom: 1rem;"></div>
                <div class="space-y-2">
                  <div *ngFor="let i of [1,2,3,4]" class="flex justify-between items-center">
                    <div class="skeleton skeleton-text" style="width: 35%;"></div>
                    <div class="skeleton skeleton-text" style="width: 45%;"></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </ng-template>
      </div>
    </div>
  `,
  styles: []
})
export class DashboardComponent implements OnInit, OnDestroy {
  systemStatus: SystemStatus | null = null
  
  // Loading states
  isLoadingSystemStatus = true;
  isLoadingHoldings = true;
  isLoadingSignals = true;
  isLoadingSymbols = true;
  isLoadingMarketOverview = true;
  recentPredictions: Prediction[] = [];
  lastUpdated = new Date();
  
  // New trading-focused data
  topHoldings: any[] = [];
  signalSummary: any = null;
  marketIndices: any[] = [];
  sectorPerformance: any[] = [];
  managedSymbols: any[] = [];

  private subscriptions: Subscription[] = [];

  constructor(private systemStatusService: SystemStatusService) {}

  ngOnInit() {
    // Subscribe to system status updates
    this.subscriptions.push(
      this.systemStatusService.systemStatus$.subscribe(status => {
        this.systemStatus = status;
        this.lastUpdated = new Date();
      })
    );

    // Subscribe to predictions updates
    this.subscriptions.push(
      this.systemStatusService.predictions$.subscribe(predictions => {
        this.recentPredictions = predictions.slice(0, 10); // Show only recent 10
      })
    );

    // Initial data load
    this.loadData();
  }

  ngOnDestroy() {
    this.subscriptions.forEach(sub => sub.unsubscribe());
  }

  loadData() {
    // Load system status
    this.isLoadingSystemStatus = true;
    this.systemStatusService.getSystemStatus().subscribe({
      next: (status) => {
        this.systemStatus = status;
        this.lastUpdated = new Date();
        this.isLoadingSystemStatus = false;
      },
      error: (error) => {
        console.error('Error loading system status:', error);
        this.isLoadingSystemStatus = false;
      }
    });

    // Load recent predictions
    this.systemStatusService.getPredictions(10).subscribe({
      next: (predictions) => {
        this.recentPredictions = predictions;
      },
      error: (error) => {
        console.error('Error loading predictions:', error);
      }
    });

    // Load trading-focused data
    this.loadTopHoldings();
    this.loadSignalSummary();
    this.loadMarketOverview();
    this.loadManagedSymbols();
  }


  loadTopHoldings() {
    this.isLoadingHoldings = true;
    console.log('Loading top holdings...');
    this.systemStatusService.getTopHoldings().subscribe({
      next: (holdings) => {
        console.log('Top holdings loaded successfully:', holdings);
        this.topHoldings = holdings;
        this.isLoadingHoldings = false;
        console.log('Holdings loading state set to false');
      },
      error: (error) => {
        console.error('Error loading top holdings:', error);
        this.isLoadingHoldings = false;
        console.log('Holdings loading state set to false (error)');
      }
    });
  }

  loadSignalSummary() {
    this.isLoadingSignals = true;
    this.systemStatusService.getSignalSummary().subscribe({
      next: (summary) => {
        this.signalSummary = summary;
        this.isLoadingSignals = false;
      },
      error: (error) => {
        console.error('Error loading signal summary:', error);
        this.isLoadingSignals = false;
      }
    });
  }

  loadMarketOverview() {
    this.isLoadingMarketOverview = true;
    this.systemStatusService.getMarketIndices().subscribe({
      next: (indices) => {
        this.marketIndices = indices;
      },
      error: (error) => {
        console.error('Error loading market indices:', error);
      }
    });

    this.systemStatusService.getSectorPerformance().subscribe({
      next: (sectors) => {
        this.sectorPerformance = sectors;
        this.isLoadingMarketOverview = false;
      },
      error: (error) => {
        console.error('Error loading sector performance:', error);
        this.isLoadingMarketOverview = false;
      }
    });
  }

  loadManagedSymbols() {
    this.isLoadingSymbols = true;
    // Get managed symbols with REAL market data from the backend
    this.systemStatusService.getManagedSymbolsWithMarketData().subscribe({
      next: (response) => {
        // Extract symbols array from response - this already has real market data!
        this.managedSymbols = response.symbols || [];
        this.isLoadingSymbols = false;
      },
      error: (error) => {
        console.error('Error loading managed symbols with market data:', error);
        // Fallback to basic symbols if market data endpoint fails
        this.systemStatusService.getSymbols().subscribe({
          next: (response) => {
            const symbols = response.symbols || response || [];
            this.managedSymbols = symbols.filter((symbol: any) => symbol.status === 'active' || symbol.status === 'monitoring');
            this.isLoadingSymbols = false;
          },
          error: (fallbackError) => {
            console.error('Error loading symbols:', fallbackError);
            this.managedSymbols = [];
            this.isLoadingSymbols = false;
          }
        });
      }
    });
  }

  getMockPrice(symbol: string): number {
    // Mock prices for different symbols
    const mockPrices: { [key: string]: number } = {
      'BTC-USD': 65000,
      'NVDA': 777.50,
      'TSLA': 245.30,
      'SOXL': 45.25,
      'CVX': 142.80,
      'MPC': 165.40,
      'RIVN': 12.85,
      'SLV': 24.15
    };
    return mockPrices[symbol] || 100;
  }

  getMockChange(symbol: string): number {
    // Mock percentage changes
    const mockChanges: { [key: string]: number } = {
      'BTC-USD': 3.85,
      'NVDA': 0.26,
      'TSLA': 1.15,
      'SOXL': 1.38,
      'CVX': -0.45,
      'MPC': 2.10,
      'RIVN': -1.25,
      'SLV': 0.75
    };
    return mockChanges[symbol] || 0;
  }

  getSignalClass(signalType: string): string {
    const baseClass = 'status-indicator';
    
    switch (signalType.toLowerCase()) {
      case 'buy':
      case 'strong_buy':
        return baseClass + ' status-active';
      case 'sell':
      case 'strong_sell':
        return baseClass + ' status-error';
      case 'hold':
        return baseClass + ' status-idle';
      default:
        return baseClass + ' status-warning';
    }
  }

  formatUptime(seconds: number | undefined): string {
    if (!seconds) return 'N/A';
    
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    } else if (minutes > 0) {
      return `${minutes}m ${secs}s`;
    } else {
      return `${secs}s`;
    }
  }

  formatPercentage(value: number | undefined): string {
    if (value === null || value === undefined) return '0';
    return (value * 100).toFixed(1);
  }

  // Helper method to access Math object in template
  get Math() {
    return Math;
  }

  getStatusClass(status: string): string {
    const baseClass = 'status-indicator';
    
    switch (status.toLowerCase()) {
      case 'active':
        return `${baseClass} status-active`;
      case 'monitoring':
        return `${baseClass} status-warning`;
      case 'watchlist':
        return `${baseClass} status-idle`;
      default:
        return `${baseClass} status-warning`;
    }
  }

}