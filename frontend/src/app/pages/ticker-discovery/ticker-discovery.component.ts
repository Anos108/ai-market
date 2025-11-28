import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { interval, Subscription } from 'rxjs';
import { ModalService } from '../../shared/modal/modal.service';

@Component({
  selector: 'app-ticker-discovery',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="min-h-screen bg-gray-50">
      <!-- Header -->
      <div class="bg-white shadow-sm border-b border-gray-200">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div class="py-6">
            <div class="flex items-center justify-between">
              <div>
                <h1 class="text-3xl font-bold text-gray-900">Ticker Discovery</h1>
                <p class="mt-2 text-sm text-gray-600">Market Scanner & Opportunity Ranker</p>
              </div>
              <div class="flex items-center space-x-4">
                <button (click)="scanMarket()" 
                        [disabled]="scanning"
                        class="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors">
                  <span *ngIf="!scanning">Scan Market</span>
                  <span *ngIf="scanning">Scanning...</span>
                </button>
                <div class="text-right">
                  <div class="text-sm text-gray-500">Last Updated</div>
                  <div class="text-sm font-medium text-gray-900">{{ lastUpdated | date:'short' }}</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Main Content -->
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        
        <!-- Summary Cards -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div class="bg-white rounded-lg shadow p-6">
            <div class="flex items-center">
              <div class="flex-shrink-0">
                <div class="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
                  <svg class="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
                  </svg>
                </div>
              </div>
              <div class="ml-4">
                <p class="text-sm font-medium text-gray-500">Total Scanned</p>
                <p class="text-2xl font-semibold text-gray-900">{{ scannerSummary?.total_scanned || 0 }}</p>
              </div>
            </div>
          </div>

          <div class="bg-white rounded-lg shadow p-6">
            <div class="flex items-center">
              <div class="flex-shrink-0">
                <div class="w-8 h-8 bg-green-100 rounded-lg flex items-center justify-center">
                  <svg class="w-5 h-5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                  </svg>
                </div>
              </div>
              <div class="ml-4">
                <p class="text-sm font-medium text-gray-500">Triggers Found</p>
                <p class="text-2xl font-semibold text-gray-900">{{ scannerSummary?.triggers_found || 0 }}</p>
              </div>
            </div>
          </div>

          <div class="bg-white rounded-lg shadow p-6">
            <div class="flex items-center">
              <div class="flex-shrink-0">
                <div class="w-8 h-8 bg-yellow-100 rounded-lg flex items-center justify-center">
                  <svg class="w-5 h-5 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
                  </svg>
                </div>
              </div>
              <div class="ml-4">
                <p class="text-sm font-medium text-gray-500">High Priority</p>
                <p class="text-2xl font-semibold text-gray-900">{{ scannerSummary?.high_priority || 0 }}</p>
              </div>
            </div>
          </div>

          <div class="bg-white rounded-lg shadow p-6">
            <div class="flex items-center">
              <div class="flex-shrink-0">
                <div class="w-8 h-8 bg-purple-100 rounded-lg flex items-center justify-center">
                  <svg class="w-5 h-5 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
                  </svg>
                </div>
              </div>
              <div class="ml-4">
                <p class="text-sm font-medium text-gray-500">Avg Confidence</p>
                <p class="text-2xl font-semibold text-gray-900">{{ (scannerSummary?.avg_confidence || 0) | number:'1.1-1' }}%</p>
              </div>
            </div>
          </div>
        </div>

        <!-- Content -->
        <div class="bg-white rounded-lg shadow p-6">
          <h3 class="text-lg font-medium text-gray-900 mb-4">Ticker Discovery System</h3>
          <p class="text-gray-600 mb-4">
            The Ticker Discovery system scans the market for trading opportunities and ranks them based on multiple criteria.
          </p>
          
          <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div class="border border-gray-200 rounded-lg p-4">
              <h4 class="font-medium text-gray-900 mb-2">Market Scanner</h4>
              <p class="text-sm text-gray-600 mb-3">
                Scans the market universe for opportunities based on various triggers like volatility, volume, momentum, and breakout patterns.
              </p>
              <div class="text-sm text-gray-500">
                <div>Total Scanned: {{ scannerSummary?.total_scanned || 0 }}</div>
                <div>Triggers Found: {{ scannerSummary?.triggers_found || 0 }}</div>
                <div>High Priority: {{ scannerSummary?.high_priority || 0 }}</div>
              </div>
            </div>
            
            <div class="border border-gray-200 rounded-lg p-4">
              <h4 class="font-medium text-gray-900 mb-2">Opportunity Ranker</h4>
              <p class="text-sm text-gray-600 mb-3">
                Ranks ticker opportunities based on multiple criteria including Sharpe ratio, confidence, risk-adjusted return, and technical strength.
              </p>
              <div class="text-sm text-gray-500">
                <div>Total Ranked: {{ rankerSummary?.total_ranked || 0 }}</div>
                <div>Avg Score: {{ (rankerSummary?.avg_score || 0) | number:'1.2-2' }}</div>
                <div>Avg Confidence: {{ (rankerSummary?.avg_confidence || 0) | number:'1.1-1' }}%</div>
              </div>
            </div>
          </div>
        </div>

        <!-- Discovered Tickers Section -->
        <div *ngIf="discoveredTickers.length > 0" class="mt-8">
          <div class="bg-white rounded-lg shadow">
            <div class="px-6 py-4 border-b border-gray-200 flex justify-between items-center">
              <div>
                <h3 class="text-lg font-medium text-gray-900">Discovered Opportunities</h3>
                <p class="text-sm text-gray-600">Tickers found during the latest scan</p>
              </div>
              <button (click)="toggleHistory()" 
                      class="bg-gray-100 hover:bg-gray-200 text-gray-700 px-3 py-2 rounded-lg text-sm font-medium transition-colors">
                <span *ngIf="!showHistory">Show History</span>
                <span *ngIf="showHistory">Hide History</span>
              </button>
            </div>
            <div class="overflow-x-auto">
              <table class="min-w-full divide-y divide-gray-200">
                <thead class="bg-gray-50">
                  <tr>
                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Symbol</th>
                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Trigger</th>
                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Priority</th>
                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Confidence</th>
                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Score</th>
                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Description</th>
                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
                  </tr>
                </thead>
                <tbody class="bg-white divide-y divide-gray-200">
                  <tr *ngFor="let ticker of discoveredTickers" class="hover:bg-gray-50">
                    <td class="px-6 py-4 whitespace-nowrap">
                      <span class="text-sm font-medium text-blue-600">{{ ticker.symbol }}</span>
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap">
                      <span class="text-sm text-gray-900 capitalize">{{ ticker.trigger }}</span>
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap">
                      <span class="inline-flex px-2 py-1 text-xs font-semibold rounded-full" 
                            [class]="'bg-' + (ticker.priority === 'high' ? 'red' : (ticker.priority === 'medium' ? 'yellow' : 'gray')) + '-100 text-' + (ticker.priority === 'high' ? 'red' : (ticker.priority === 'medium' ? 'yellow' : 'gray')) + '-800'">
                        {{ ticker.priority }}
                      </span>
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {{ (ticker.confidence * 100) | number:'1.0-0' }}%
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {{ ticker.score }}
                    </td>
                    <td class="px-6 py-4">
                      <div class="text-sm text-gray-900">{{ ticker.description }}</div>
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap">
                            <button (click)="addToPortfolio(ticker)"
                                      [disabled]="addingToPortfolio[ticker.symbol] || isInPortfolio(ticker.symbol)"
                                      class="px-3 py-1 rounded-md text-xs font-medium transition-colors"
                                      [class]="isInPortfolio(ticker.symbol) 
                                                  ? 'bg-gray-100 text-gray-500 cursor-not-allowed' 
                                                  : addingToPortfolio[ticker.symbol]
                                                    ? 'bg-blue-200 text-blue-600 cursor-not-allowed'
                                                    : 'bg-blue-100 text-blue-600 hover:bg-blue-200'">
                                <span *ngIf="addingToPortfolio[ticker.symbol]">Adding...</span>
                                <span *ngIf="!addingToPortfolio[ticker.symbol] && !isInPortfolio(ticker.symbol)">Add to Symbol Mgmt</span>
                                <span *ngIf="isInPortfolio(ticker.symbol)">✓ In Portfolio</span>
                              </button>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <!-- History Section -->
        <div *ngIf="showHistory && scanHistory.length > 0" class="mt-8">
          <div class="bg-white rounded-lg shadow">
            <div class="px-6 py-4 border-b border-gray-200">
              <h3 class="text-lg font-medium text-gray-900">Discovery History</h3>
              <p class="text-sm text-gray-600">Previous ticker discovery scan results by date</p>
            </div>
            <div class="overflow-x-auto">
              <table class="min-w-full divide-y divide-gray-200">
                <thead class="bg-gray-50">
                  <tr>
                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Date/Time</th>
                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Total Scanned</th>
                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Triggers Found</th>
                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">High Priority</th>
                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Avg Confidence</th>
                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                  </tr>
                </thead>
                <tbody class="bg-white divide-y divide-gray-200">
                  <tr *ngFor="let scan of scanHistory" class="hover:bg-gray-50">
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {{ scan.timestamp | date:'MMM d, y, h:mm:ss a' }}
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {{ scan.total_scanned }}
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {{ scan.triggers_found }}
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {{ scan.high_priority }}
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {{ scan.avg_confidence | number:'1.1-1' }}%
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap">
                      <span class="inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-green-100 text-green-800">
                        {{ scan.status || 'Completed' }}
                      </span>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <!-- Loading State -->
        <div *ngIf="loading" class="flex items-center justify-center py-12">
          <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <span class="ml-3 text-gray-600">Loading ticker discovery data...</span>
        </div>

        <!-- Error State -->
        <div *ngIf="error" class="bg-red-50 border border-red-200 rounded-lg p-4">
          <div class="flex">
            <div class="flex-shrink-0">
              <svg class="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" />
              </svg>
            </div>
            <div class="ml-3">
              <h3 class="text-sm font-medium text-red-800">Error loading data</h3>
              <div class="mt-2 text-sm text-red-700">{{ error }}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `,
  styles: []
})
export class TickerDiscoveryComponent implements OnInit, OnDestroy {
  scannerSummary: any = null;
  rankerSummary: any = null;
  discoveredTickers: any[] = [];
  loading = true;
  scanning = false;
  error: string | null = null;
  lastUpdated: Date = new Date();
  addingToPortfolio: { [key: string]: boolean } = {};
  portfolioSymbols: string[] = [];
  showHistory: boolean = false;
  scanHistory: any[] = [];
  
  private refreshSubscription: Subscription = new Subscription();

  constructor(private http: HttpClient, private modalService: ModalService) {}

  ngOnInit() {
    this.loadData();
    this.loadDiscoveredTickers();
    this.loadPortfolioSymbols();
    
    // Refresh data every 30 seconds
    this.refreshSubscription = interval(30000).subscribe(() => {
      this.loadData();
      this.loadDiscoveredTickers();
      this.loadPortfolioSymbols();
    });
  }

  ngOnDestroy() {
    this.refreshSubscription.unsubscribe();
  }

  loadData() {
    this.loading = true;
    this.error = null;

    // Load scanner summary
    this.http.get<any>('http://localhost:8001/ticker-discovery/scanner-summary').subscribe({
      next: (data) => {
        this.scannerSummary = data;
        this.lastUpdated = new Date();
      },
      error: (err) => {
        console.error('Error loading scanner summary:', err);
        this.error = 'Failed to load scanner summary';
      }
    });

    // Load ranker summary
    this.http.get<any>('http://localhost:8001/ticker-discovery/ranker-summary').subscribe({
      next: (data) => {
        this.rankerSummary = data;
        this.loading = false;
      },
      error: (err) => {
        console.error('Error loading ranker summary:', err);
        this.error = 'Failed to load ranker summary';
        this.loading = false;
      }
    });
  }

  scanMarket() {
    this.scanning = true;
    this.error = null; // Clear any previous errors
    this.http.post<any>('http://localhost:8001/ticker-discovery/scan-market', {}).subscribe({
      next: (data) => {
        this.scanning = false;
        this.lastUpdated = new Date();
        // Reload data after scan including discovered tickers
        this.loadData();
        this.loadDiscoveredTickers();
      },
      error: (err) => {
        console.error('Error scanning market:', err);
        this.scanning = false;
        this.error = 'Failed to scan market';
      }
    });
  }

  loadDiscoveredTickers() {
    this.http.get<any>('http://localhost:8001/ticker-discovery/scan-details').subscribe({
      next: (data) => {
        this.discoveredTickers = data.discovered_tickers || [];
      },
      error: (err) => {
        console.error('Error loading discovered tickers:', err);
      }
    });
  }

  loadPortfolioSymbols() {
    this.http.get<any>('http://localhost:8001/api/symbols').subscribe({
      next: (response) => {
        const symbols = response.symbols || [];
        this.portfolioSymbols = symbols.map((s: any) => s.symbol?.toUpperCase()).filter(Boolean);
      },
      error: (err) => {
        console.error('Error loading portfolio symbols:', err);
      }
    });
  }

  isInPortfolio(symbol: string): boolean {
    return this.portfolioSymbols.includes(symbol?.toUpperCase());
  }

  addToPortfolio(ticker: any) {
    const symbol = ticker.symbol?.toUpperCase();
    if (!symbol || this.isInPortfolio(symbol) || this.addingToPortfolio[symbol]) {
      return;
    }

    this.addingToPortfolio[symbol] = true;
    
    const symbolData = {
      symbol: symbol,
      name: this.getSymbolName(ticker),
      sector: ticker.sector || 'Technology',
      industry: ticker.industry || 'General'
    };

    this.http.post<any>('http://localhost:8001/symbols/add-from-discovery', symbolData).subscribe({
      next: (response) => {
        this.addingToPortfolio[symbol] = false;
        if (response.success) {
          // Add to portfolio symbols list
          this.portfolioSymbols.push(symbol);
          console.log(`✅ Successfully added ${symbol} to symbol management`);
          // Show user confirmation with modal
          const message = `${symbol} added to Symbol Management. Check Portfolio Management page to see it.`;
          this.modalService.success(message, 'Symbol Added');
        }
      },
      error: (err) => {
        console.error(`Error adding ${symbol} to portfolio:`, err);
        this.addingToPortfolio[symbol] = false;
        const errorMessage = `Failed to add ${symbol} to portfolio: ${err.error?.detail || 'Unknown error'}`;
        this.modalService.error(errorMessage, 'Failed to Add Symbol');
      }
    });
  }

  private getSymbolName(ticker: any): string {
    return ticker.name || ticker.symbol || `${ticker.symbol} Corporation`;
  }

  toggleHistory() {
    this.showHistory = !this.showHistory;
    if (this.showHistory && this.scanHistory.length === 0) {
      this.loadScanHistory();
    }
  }

  loadScanHistory() {
    this.http.get<any>('http://localhost:8001/ticker-discovery/history').subscribe({
      next: (history) => {
        this.scanHistory = history || [];
        console.log('📋 Loaded ticker discovery history:', this.scanHistory);
      },
      error: (err) => {
        console.error('Error loading scan history:', err);
      }
    });
  }
}
