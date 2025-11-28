import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { interval, Subscription, forkJoin } from 'rxjs';
import { switchMap, catchError } from 'rxjs/operators';
import { of } from 'rxjs';
import { SkeletonComponent } from '../../shared/skeleton/skeleton.component';
import { SystemStatusService } from '../../services/system-status.service';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';

interface DayForecast {
  symbol: string;
  horizon: string;
  predicted_price: number;
  current_price?: number;
  confidence: number;
  direction: string;
  signal_strength: string;
  market_regime: string;
  technical_indicators: TechnicalIndicator[];
  volatility_forecast: number;
  volume_forecast: number;
  risk_score: number;
  created_at: string;
  valid_until: string;
}

interface SwingForecast {
  symbol: string;
  horizon: string;
  predicted_price: number;
  current_price?: number;
  confidence: number;
  direction: string;
  signal_strength: string;
  market_regime: string;
  key_events: MarketEvent[];
  macro_factors: MacroFactor[];
  technical_score: number;
  fundamental_score: number;
  sentiment_score: number;
  risk_score: number;
  target_price: number;
  stop_loss: number;
  created_at: string;
  valid_until: string;
}

interface TechnicalIndicator {
  name: string;
  value: number;
  signal: string;
  strength: number;
  timestamp: string;
}

interface MarketEvent {
  event_id: string;
  event_type: string;
  symbol: string;
  impact: string;
  expected_date: string;
  description: string;
  historical_impact: number;
  confidence: number;
}

interface MacroFactor {
  name: string;
  value: number;
  previous_value: number;
  change: number;
  change_percent: number;
  impact_score: number;
  last_updated: string;
}

interface ForecastComparison {
  symbol: string;
  day_forecast: any;
  swing_forecast: any;
  comparison: {
    direction_agreement: boolean;
    confidence_difference: number;
    price_difference: number;
    risk_difference: number;
    recommended_strategy: string;
  };
  generated_at: string;
}

interface ForecastSummary {
  agent_type: string;
  forecast_horizons: string[];
  active_forecasts: number;
  total_forecasts: number;
  recent_forecasts: any[];
  metrics: any;
  last_updated: string;
  market_events?: number;
  macro_indicators?: number;
}

@Component({
  selector: 'app-forecasting-dashboard',
  standalone: true,
  imports: [CommonModule, FormsModule, SkeletonComponent],
  templateUrl: './forecasting-dashboard.component.html',
  styleUrls: ['./forecasting-dashboard.component.css']
})
export class ForecastingDashboardComponent implements OnInit, OnDestroy {
  // Data properties
  dayForecasts: DayForecast[] = [];
  swingForecasts: SwingForecast[] = [];
  forecastComparison: ForecastComparison | null = null;
  dayForecastSummary: ForecastSummary | null = null;
  swingForecastSummary: ForecastSummary | null = null;
  
  // RAG Analysis data
  latestRAGAnalysis: any = null;
  expandedRAGSectors: Set<string> = new Set();
  
  // API configuration
  apiUrl = 'http://localhost:8001';
  
  // UI state
  activeTab = 'day-forecasts';
  isLoading = false;
  error: string | null = null;
  autoRefresh = true;
  refreshInterval = 60000; // 1 minute
  
  // Loading states
  isLoadingSummary = true;
  isLoadingDayForecasts = true;
  isLoadingSwingForecasts = true;
  isLoadingComparison = true;
  isLoadingRAGAnalysis = true;
  
  // Form data
  selectedSymbol = 'BTC-USD';
  selectedDayHorizon = 'end_of_day';
  selectedSwingHorizon = 'medium_swing';
  
  // Available options - will be populated from managed symbols
  symbols: string[] = [];
  
  // Filter options
  selectedDayFilter = 'all';
  selectedSwingFilter = 'all';
  filterOptions = [
    { value: 'all', label: 'All Signals' },
    { value: 'BUY', label: 'BUY' },
    { value: 'HOLD', label: 'HOLD' },
    { value: 'SELL', label: 'SELL' }
  ];
  dayHorizons = [
    { value: 'intraday', label: 'Intraday (1-4 hours)' },
    { value: 'end_of_day', label: 'End of Day' },
    { value: 'next_day_open', label: 'Next Day Open' },
    { value: 'next_day_close', label: 'Next Day Close' }
  ];
  swingHorizons = [
    { value: 'short_swing', label: 'Short Swing (3-5 days)' },
    { value: 'medium_swing', label: 'Medium Swing (5-7 days)' },
    { value: 'long_swing', label: 'Long Swing (7-10 days)' }
  ];
  
  // Advanced Forecasting
  advancedForecasts: any[] = [];
  isLoadingAdvanced = false;
  selectedAdvancedFilter = 'all';
  
  // Expand/Collapse state
  expandedDayForecasts: Set<string> = new Set();
  expandedSwingForecasts: Set<string> = new Set();
  expandedAdvancedForecasts: Set<string> = new Set();
  
  // Sorting state
  daySortColumn: string = 'symbol';
  daySortDirection: 'asc' | 'desc' = 'asc';
  swingSortColumn: string = 'symbol';
  swingSortDirection: 'asc' | 'desc' = 'asc';
  advancedSortColumn: string = 'symbol';
  advancedSortDirection: 'asc' | 'desc' = 'asc';
  
  // Subscriptions
  private refreshSubscription?: Subscription;
  
  constructor(
    private http: HttpClient, 
    private systemStatusService: SystemStatusService,
    private sanitizer: DomSanitizer
  ) {}
  
  ngOnInit(): void {
    this.loadManagedSymbols();
    this.loadAllData();
    this.loadGeneratedForecasts();
    this.loadLatestRAGAnalysis();
    this.startAutoRefresh();
  }
  
  ngOnDestroy(): void {
    if (this.refreshSubscription) {
      this.refreshSubscription.unsubscribe();
    }
  }

  toggleDayForecastExpansion(symbol: string): void {
    if (this.expandedDayForecasts.has(symbol)) {
      this.expandedDayForecasts.delete(symbol);
    } else {
      this.expandedDayForecasts.add(symbol);
    }
  }

  toggleSwingForecastExpansion(symbol: string): void {
    if (this.expandedSwingForecasts.has(symbol)) {
      this.expandedSwingForecasts.delete(symbol);
    } else {
      this.expandedSwingForecasts.add(symbol);
    }
  }

  toggleAdvancedForecastExpansion(symbol: string): void {
    if (this.expandedAdvancedForecasts.has(symbol)) {
      this.expandedAdvancedForecasts.delete(symbol);
    } else {
      this.expandedAdvancedForecasts.add(symbol);
    }
  }

  isDayForecastExpanded(symbol: string): boolean {
    return this.expandedDayForecasts.has(symbol);
  }

  isSwingForecastExpanded(symbol: string): boolean {
    return this.expandedSwingForecasts.has(symbol);
  }

  isAdvancedForecastExpanded(symbol: string): boolean {
    return this.expandedAdvancedForecasts.has(symbol);
  }

  trackBySymbol(index: number, forecast: any): string {
    return forecast.symbol;
  }

  sortDayForecasts(column: string): void {
    if (this.daySortColumn === column) {
      this.daySortDirection = this.daySortDirection === 'asc' ? 'desc' : 'asc';
    } else {
      this.daySortColumn = column;
      this.daySortDirection = 'asc';
    }
    this.dayForecasts.sort((a, b) => this.compareValues(a, b, column, this.daySortDirection));
  }

  sortSwingForecasts(column: string): void {
    if (this.swingSortColumn === column) {
      this.swingSortDirection = this.swingSortDirection === 'asc' ? 'desc' : 'asc';
    } else {
      this.swingSortColumn = column;
      this.swingSortDirection = 'asc';
    }
    this.swingForecasts.sort((a, b) => this.compareValues(a, b, column, this.swingSortDirection));
  }

  sortAdvancedForecasts(column: string): void {
    if (this.advancedSortColumn === column) {
      this.advancedSortDirection = this.advancedSortDirection === 'asc' ? 'desc' : 'asc';
    } else {
      this.advancedSortColumn = column;
      this.advancedSortDirection = 'asc';
    }
    this.advancedForecasts.sort((a, b) => this.compareValues(a, b, column, this.advancedSortDirection));
  }

  private compareValues(a: any, b: any, column: string, direction: 'asc' | 'desc'): number {
    let aValue: any;
    let bValue: any;

    // Map column names to property names
    switch (column) {
      case 'symbol':
        aValue = a.symbol;
        bValue = b.symbol;
        break;
      case 'signal':
        aValue = a.direction || a.final_signal;
        bValue = b.direction || b.final_signal;
        break;
      case 'confidence':
        aValue = a.confidence || a.final_confidence;
        bValue = b.confidence || b.final_confidence;
        break;
      case 'predicted_price':
        aValue = a.predicted_price;
        bValue = b.predicted_price;
        break;
      case 'target_price':
        aValue = a.target_price;
        bValue = b.target_price;
        break;
      case 'risk':
        aValue = a.risk_score;
        bValue = b.risk_score;
        break;
      case 'valid_until':
        aValue = a.valid_until ? new Date(a.valid_until).getTime() : 0;
        bValue = b.valid_until ? new Date(b.valid_until).getTime() : 0;
        break;
      case 'market_regime':
        aValue = a.market_regime;
        bValue = b.market_regime;
        break;
      case 'agents':
        aValue = a.total_agents_contributing || 0;
        bValue = b.total_agents_contributing || 0;
        break;
      default:
        aValue = a[column];
        bValue = b[column];
    }

    // Handle null/undefined values
    if (aValue === null || aValue === undefined) aValue = '';
    if (bValue === null || bValue === undefined) bValue = '';

    // Compare values
    let comparison = 0;
    if (typeof aValue === 'string' && typeof bValue === 'string') {
      comparison = aValue.localeCompare(bValue);
    } else {
      comparison = aValue > bValue ? 1 : aValue < bValue ? -1 : 0;
    }

    return direction === 'asc' ? comparison : -comparison;
  }

  getSortIcon(column: string, sortColumn: string, sortDirection: 'asc' | 'desc'): string {
    if (column !== sortColumn) {
      return '';
    }
    return sortDirection === 'asc' ? '▲' : '▼';
  }

  private parseTechnicalIndicators(indicators: any): TechnicalIndicator[] {
    if (!indicators) return [];
    
    // If it's already an array, return it
    if (Array.isArray(indicators)) return indicators;
    
    // If it's a string, try to parse it
    if (typeof indicators === 'string') {
      try {
        const parsed = JSON.parse(indicators);
        
        // If it's the old dictionary format, convert to array format
        if (typeof parsed === 'object' && !Array.isArray(parsed)) {
          return this.convertDictToArray(parsed);
        }
        
        // If it's already an array, return it
        if (Array.isArray(parsed)) return parsed;
        
        return [];
      } catch (e) {
        console.error('Error parsing technical indicators:', e);
        return [];
      }
    }
    
    return [];
  }

  private convertDictToArray(dict: any): TechnicalIndicator[] {
    const indicators: TechnicalIndicator[] = [];
    
    // Convert old dictionary format to array format
    if (dict.rsi !== undefined) {
      indicators.push({
        name: 'RSI',
        value: dict.rsi,
        signal: dict.rsi < 30 ? 'buy' : dict.rsi > 70 ? 'sell' : 'hold',
        strength: Math.abs(dict.rsi - 50) / 50,
        timestamp: new Date().toISOString()
      });
    }
    
    if (dict.macd !== undefined) {
      indicators.push({
        name: 'MACD',
        value: dict.macd,
        signal: dict.macd > 0 ? 'buy' : dict.macd < -0.5 ? 'sell' : 'hold',
        strength: Math.min(1.0, Math.abs(dict.macd) * 2),
        timestamp: new Date().toISOString()
      });
    }
    
    if (dict.bollinger_position !== undefined) {
      indicators.push({
        name: 'Bollinger',
        value: dict.bollinger_position,
        signal: dict.bollinger_position < 0.2 ? 'buy' : dict.bollinger_position > 0.8 ? 'sell' : 'hold',
        strength: Math.abs(dict.bollinger_position - 0.5) * 2,
        timestamp: new Date().toISOString()
      });
    }
    
    if (dict.volume_trend !== undefined) {
      indicators.push({
        name: 'Volume',
        value: dict.volume_trend === 'increasing' ? 1.0 : 0.0,
        signal: dict.volume_trend === 'increasing' ? 'buy' : 'sell',
        strength: 0.7,
        timestamp: new Date().toISOString()
      });
    }
    
    return indicators;
  }

  private parseJsonField(field: any, defaultValue: any = null): any {
    if (!field) return defaultValue;
    
    if (typeof field === 'string') {
      try {
        return JSON.parse(field);
      } catch (e) {
        console.error('Error parsing JSON field:', e);
        return defaultValue;
      }
    }
    
    return field;
  }
  
  loadManagedSymbols(): void {
    this.http.get<any>('http://localhost:8001/symbols/managed-with-market-data')
      .pipe(
        catchError(error => {
          console.error('Error loading managed symbols:', error);
          // Fallback to default symbols if API fails
          this.symbols = ['BTC-USD', 'SOXL', 'NVDA', 'RIVN', 'TSLA', 'SPY', 'META', 'AMD', 'INTC', 'TQQQ'];
          return of(null);
        })
      )
      .subscribe(response => {
        if (response && response.symbols) {
          this.symbols = response.symbols
            .map((symbol: any) => symbol.symbol);
          
          // Set default selected symbol to first available symbol
          if (this.symbols.length > 0 && !this.symbols.includes(this.selectedSymbol)) {
            this.selectedSymbol = this.symbols[0];
          }
          
          console.log('Loaded managed symbols for forecasting:', this.symbols);
        }
      });
  }

  startAutoRefresh(): void {
    if (this.autoRefresh) {
      this.refreshSubscription = interval(this.refreshInterval)
        .subscribe(() => {
          this.loadAllData();
        });
    }
  }
  
  stopAutoRefresh(): void {
    if (this.refreshSubscription) {
      this.refreshSubscription.unsubscribe();
    }
  }
  
  toggleAutoRefresh(): void {
    this.autoRefresh = !this.autoRefresh;
    if (this.autoRefresh) {
      this.startAutoRefresh();
    } else {
      this.stopAutoRefresh();
    }
  }
  
  setActiveTab(tab: string): void {
    this.activeTab = tab;
  }
  
  loadAllData(): void {
    this.isLoading = true;
    this.error = null;
    this.isLoadingSummary = true;
    
    // Load forecast summaries
    this.http.get<ForecastSummary>('http://localhost:8001/forecasting/day-forecast/summary')
      .pipe(catchError(error => {
        console.error('Error loading day forecast summary:', error);
        return of(null);
      }))
      .subscribe(data => {
        this.dayForecastSummary = data;
      });
    
    this.http.get<ForecastSummary>('http://localhost:8001/forecasting/swing-forecast/summary')
      .pipe(catchError(error => {
        console.error('Error loading swing forecast summary:', error);
        return of(null);
      }))
      .subscribe(data => {
        this.swingForecastSummary = data;
        this.isLoading = false;
        this.isLoadingSummary = false;
      });
    
    // Also reload generated forecasts
    this.loadGeneratedForecasts();
  }

  loadLatestRAGAnalysis(): void {
    this.isLoadingRAGAnalysis = true;
    this.systemStatusService.getLatestRAGAnalysis().subscribe({
      next: (data) => {
        this.latestRAGAnalysis = data;
        this.isLoadingRAGAnalysis = false;
      },
      error: (error) => {
        console.error('Error loading latest RAG analysis:', error);
        this.isLoadingRAGAnalysis = false;
      }
    });
  }
  
  loadGeneratedForecasts(): void {
    // Load day forecasts from database
    this.http.get<any[]>(`${this.apiUrl}/forecasting/day-forecasts`)
      .pipe(
        catchError(error => {
          console.error('Error loading day forecasts from database:', error);
          return of([]);
        })
      )
      .subscribe(forecasts => {
        console.log('Loaded day forecasts:', forecasts.length, forecasts);
        this.dayForecasts = forecasts.map(forecast => ({
          ...forecast,
          technical_indicators: this.parseTechnicalIndicators(forecast.technical_indicators)
        }));
        console.log('Processed day forecasts:', this.dayForecasts.length, this.dayForecasts);
        this.isLoadingDayForecasts = false;
      });
    
    // Load swing forecasts from database
    this.http.get<any[]>(`${this.apiUrl}/forecasting/swing-forecasts`)
      .pipe(
        catchError(error => {
          console.error('Error loading swing forecasts from database:', error);
          return of([]);
        })
      )
      .subscribe(forecasts => {
        this.swingForecasts = forecasts.map(forecast => ({
          ...forecast,
          technical_indicators: this.parseTechnicalIndicators(forecast.technical_indicators),
          key_events: this.parseJsonField(forecast.key_events),
          macro_factors: this.parseJsonField(forecast.macro_factors)
        }));
        this.isLoadingSwingForecasts = false;
      });
    
    // Load advanced forecasts from database
    this.http.get<any>(`${this.apiUrl}/forecasting/advanced-forecasts`)
      .pipe(
        catchError(error => {
          console.error('Error loading advanced forecasts from database:', error);
          return of([]);
        })
      )
      .subscribe(response => {
        // Handle both live generation response and saved database response
        let forecasts = [];
        if (response && response.forecasts && Array.isArray(response.forecasts)) {
          // Live generation response
          forecasts = response.forecasts;
        } else if (Array.isArray(response)) {
          // Saved database response
          forecasts = response;
        } else {
          console.error('Unexpected response format for advanced forecasts:', response);
          forecasts = [];
        }
        
        // Parse JSON fields that come as strings from the database
        this.advancedForecasts = forecasts.map((forecast: any) => ({
          ...forecast,
          agent_contributions: this.parseJsonField(forecast.agent_contributions, []),
          latent_patterns: this.parseJsonField(forecast.latent_patterns, []),
          signal_distribution: this.parseJsonField(forecast.signal_distribution, {}),
          ensemble_signal: this.parseJsonField(forecast.ensemble_signal, null),
          rag_analysis: this.parseJsonField(forecast.rag_analysis, null),
          rl_recommendation: this.parseJsonField(forecast.rl_recommendation, null),
          meta_evaluation: this.parseJsonField(forecast.meta_evaluation, null)
        }));
        console.log('Loaded advanced forecasts:', this.advancedForecasts.length, 'records');
        console.log('Sample forecast:', this.advancedForecasts[0]);
      });
    
    // Load forecast comparison from localStorage
    const savedComparison = localStorage.getItem('forecastComparison');
    if (savedComparison) {
      try {
        this.forecastComparison = JSON.parse(savedComparison);
      } catch (e) {
        console.error('Error parsing saved forecast comparison:', e);
      }
    }
    // Set loading to false after attempting to load
    this.isLoadingComparison = false;
  }
  
  saveGeneratedForecasts(): void {
    // Save day forecasts to database
    if (this.dayForecasts.length > 0) {
      this.http.post(`${this.apiUrl}/forecasting/day-forecasts/save`, this.dayForecasts)
        .pipe(
          catchError(error => {
            console.error('Error saving day forecasts to database:', error);
            return of(null);
          })
        )
        .subscribe(response => {
          if (response) {
            console.log('Day forecasts saved to database:', response);
          }
        });
    }
    
    // Save swing forecasts to database
    if (this.swingForecasts.length > 0) {
      this.http.post(`${this.apiUrl}/forecasting/swing-forecasts/save`, this.swingForecasts)
        .pipe(
          catchError(error => {
            console.error('Error saving swing forecasts to database:', error);
            return of(null);
          })
        )
        .subscribe(response => {
          if (response) {
            console.log('Swing forecasts saved to database:', response);
          }
        });
    }
    
    // Still save forecast comparison to localStorage (for quick access)
    if (this.forecastComparison) {
      localStorage.setItem('forecastComparison', JSON.stringify(this.forecastComparison));
    }
  }

  getSignalStrength(confidence: number): string {
    if (confidence >= 0.8) return 'Strong';
    if (confidence >= 0.6) return 'Medium';
    if (confidence >= 0.4) return 'Weak';
    return 'Very Weak';
  }

  getPriceChange(currentPrice: number, predictedPrice: number): number {
    if (!currentPrice || !predictedPrice) return 0;
    return ((predictedPrice - currentPrice) / currentPrice) * 100;
  }

  get filteredDayForecasts(): DayForecast[] {
    console.log('filteredDayForecasts called - filter:', this.selectedDayFilter, 'total forecasts:', this.dayForecasts.length);
    if (this.selectedDayFilter === 'all') {
      console.log('Returning all forecasts:', this.dayForecasts.length);
      return this.dayForecasts;
    }
    const filtered = this.dayForecasts.filter(forecast => forecast.direction.toUpperCase() === this.selectedDayFilter);
    console.log('Filtered forecasts:', filtered.length);
    return filtered;
  }

  get filteredSwingForecasts(): SwingForecast[] {
    if (this.selectedSwingFilter === 'all') {
      return this.swingForecasts;
    }
    return this.swingForecasts.filter(forecast => forecast.direction.toUpperCase() === this.selectedSwingFilter);
  }

  get filteredAdvancedForecasts(): any[] {
    if (this.selectedAdvancedFilter === 'all') {
      return this.advancedForecasts;
    }
    return this.advancedForecasts.filter(forecast => forecast.final_signal === this.selectedAdvancedFilter);
  }


  getSignalCount(forecasts: any[], signal: string): number {
    console.log('getSignalCount called with:', signal, 'forecasts:', forecasts?.length);
    if (!forecasts || forecasts.length === 0) {
      console.log('getSignalCount: No forecasts or empty array');
      return 0;
    }
    const count = forecasts.filter(forecast => 
      (forecast.direction?.toUpperCase() === signal) || (forecast.final_signal === signal)
    ).length;
    console.log(`getSignalCount: ${signal} signals = ${count} (total forecasts: ${forecasts.length})`);
    console.log('Sample forecast directions:', forecasts.slice(0, 3).map(f => f.direction));
    return count;
  }
  
  generateForecasts(): void {
    this.isLoading = true;
    this.isLoadingDayForecasts = true;
    this.isLoadingSwingForecasts = true;
    this.error = null;
    
    // Generate both day and swing forecasts
    const dayForecast$ = this.http.get<any>(`http://localhost:8001/forecasting/day-forecast?symbol=${this.selectedSymbol}&horizon=${this.selectedDayHorizon}`);
    const swingForecast$ = this.http.get<any>(`http://localhost:8001/forecasting/swing-forecast?symbol=${this.selectedSymbol}&horizon=${this.selectedSwingHorizon}`);
    
    // Wait for both forecasts to complete
    forkJoin([dayForecast$, swingForecast$])
      .pipe(catchError(error => {
        console.error('Error generating forecasts:', error);
        return of([null, null]);
      }))
      .subscribe(([dayResult, swingResult]) => {
        this.isLoading = false;
        this.isLoadingDayForecasts = false;
        this.isLoadingSwingForecasts = false;
        
        if (dayResult) {
          this.dayForecasts = [dayResult];
        }
        if (swingResult) {
          this.swingForecasts = [swingResult];
        }
        
        // Save the generated forecasts
        this.saveGeneratedForecasts();
        
        // Load comparison
        this.loadForecastComparison();
      });
  }

  generateForecastsForAllSymbols(): void {
    this.isLoading = true;
    this.isLoadingDayForecasts = true;
    this.isLoadingSwingForecasts = true;
    this.error = null;
    
    console.log('Generating forecasts for all managed symbols...');
    
    this.http.get<any>('http://localhost:8001/forecasting/generate-all-forecasts')
      .pipe(
        catchError(error => {
          console.error('Error generating forecasts for all symbols:', error);
          this.error = 'Failed to generate forecasts for all symbols';
          this.isLoading = false;
          this.isLoadingDayForecasts = false;
          this.isLoadingSwingForecasts = false;
          return of(null);
        })
      )
      .subscribe(response => {
        this.isLoading = false;
        this.isLoadingDayForecasts = false;
        this.isLoadingSwingForecasts = false;
        
        if (response) {
          console.log('Generated forecasts for all managed symbols:', response);
          
          // Extract day and swing forecasts from the response
          const dayForecasts: DayForecast[] = [];
          const swingForecasts: SwingForecast[] = [];
          
          response.results.forEach((result: any) => {
            if (result.status === 'success') {
              // Add day forecast
              if (result.day_forecast) {
                const dayForecast = result.day_forecast;
                dayForecasts.push({
                  symbol: dayForecast.symbol,
                  horizon: dayForecast.horizon,
                  predicted_price: dayForecast.target_price,
                  current_price: dayForecast.current_price,
                  confidence: dayForecast.confidence,
                  direction: dayForecast.signal_type.toUpperCase(),
                  signal_strength: this.getSignalStrength(dayForecast.confidence),
                  market_regime: 'normal', // Default value
                  technical_indicators: [
                    {
                      name: 'RSI',
                      value: 50 + (Math.random() - 0.5) * 20, // Mock RSI
                      signal: dayForecast.signal_type,
                      strength: dayForecast.confidence,
                      timestamp: dayForecast.timestamp
                    }
                  ],
                  volatility_forecast: Math.random() * 0.3 + 0.1, // Mock volatility
                  volume_forecast: Math.random() * 1000000 + 500000, // Mock volume
                  risk_score: Math.random() * 0.5 + 0.2, // Mock risk score
                  created_at: dayForecast.timestamp,
                  valid_until: new Date(new Date(dayForecast.timestamp).getTime() + 24 * 60 * 60 * 1000).toISOString() // 24 hours later
                });
              }
              
              // Add swing forecast
              if (result.swing_forecast) {
                const swingForecast = result.swing_forecast;
                swingForecasts.push({
                  symbol: swingForecast.symbol,
                  horizon: swingForecast.horizon,
                  predicted_price: swingForecast.target_price,
                  current_price: swingForecast.current_price,
                  confidence: swingForecast.confidence,
                  direction: swingForecast.signal_type.toUpperCase(),
                  signal_strength: this.getSignalStrength(swingForecast.confidence),
                  market_regime: swingForecast.trend || 'normal',
                  key_events: [], // Empty for now
                  macro_factors: [], // Empty for now
                  technical_score: swingForecast.confidence,
                  fundamental_score: Math.random() * 0.4 + 0.3, // Mock fundamental score
                  sentiment_score: Math.random() * 0.4 + 0.3, // Mock sentiment score
                  risk_score: Math.random() * 0.4 + 0.2, // Mock risk score
                  target_price: swingForecast.target_price,
                  stop_loss: swingForecast.stop_loss,
                  created_at: swingForecast.timestamp,
                  valid_until: new Date(new Date(swingForecast.timestamp).getTime() + swingForecast.days * 24 * 60 * 60 * 1000).toISOString() // Based on days
                });
              }
            }
          });
          
          // Update the arrays
          this.dayForecasts = dayForecasts;
          this.swingForecasts = swingForecasts;
          
          // Save to localStorage so they persist
          this.saveGeneratedForecasts();
          
          // Show success message
          this.error = null;
          
          // Refresh the summary data
          this.loadAllData();
          
          // Load comparison for the selected symbol
          this.loadForecastComparison();
          
          // Show a temporary success message
          const originalError = this.error;
          this.error = `Successfully generated ${response.forecasts_generated} forecasts for ${response.symbols_processed} symbols!`;
          setTimeout(() => {
            this.error = originalError;
          }, 5000);
        }
      });
  }
  
  loadForecastComparison(): void {
    this.isLoadingComparison = true;
    this.http.get<ForecastComparison>(`http://localhost:8001/forecasting/compare-forecasts?symbol=${this.selectedSymbol}`)
      .pipe(catchError(error => {
        console.error('Error loading forecast comparison:', error);
        return of(null);
      }))
      .subscribe(data => {
        this.forecastComparison = data;
        this.isLoadingComparison = false;
        // Save the comparison data
        this.saveGeneratedForecasts();
      });
  }
  
  getDayForecast(symbol: string, horizon: string): void {
    this.isLoading = true;
    this.isLoadingDayForecasts = true;
    this.error = null;
    
    this.http.get<DayForecast>(`http://localhost:8001/forecasting/day-forecast?symbol=${symbol}&horizon=${horizon}`)
      .pipe(catchError(error => {
        console.error('Error getting day forecast:', error);
        return of(null);
      }))
      .subscribe(data => {
        this.isLoading = false;
        this.isLoadingDayForecasts = false;
        if (data) {
          this.dayForecasts = [data];
          // Save the generated forecast
          this.saveGeneratedForecasts();
        } else {
          this.error = 'Failed to get day forecast';
        }
      });
  }
  
  getSwingForecast(symbol: string, horizon: string): void {
    this.isLoading = true;
    this.isLoadingSwingForecasts = true;
    this.error = null;
    
    this.http.get<SwingForecast>(`http://localhost:8001/forecasting/swing-forecast?symbol=${symbol}&horizon=${horizon}`)
      .pipe(catchError(error => {
        console.error('Error getting swing forecast:', error);
        return of(null);
      }))
      .subscribe(data => {
        this.isLoading = false;
        this.isLoadingSwingForecasts = false;
        if (data) {
          this.swingForecasts = [data];
          // Save the generated forecast
          this.saveGeneratedForecasts();
        } else {
          this.error = 'Failed to get swing forecast';
        }
      });
  }
  
  getDirectionColor(direction: string): string {
    switch (direction.toLowerCase()) {
      case 'up': return 'text-green-600 bg-green-100';
      case 'down': return 'text-red-600 bg-red-100';
      case 'sideways': return 'text-yellow-600 bg-yellow-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  }
  
  getSignalStrengthColor(strength: string): string {
    switch (strength.toLowerCase()) {
      case 'very_strong': return 'text-green-800 bg-green-200';
      case 'strong': return 'text-green-700 bg-green-100';
      case 'moderate': return 'text-yellow-700 bg-yellow-100';
      case 'weak': return 'text-gray-700 bg-gray-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  }
  
  getConfidenceColor(confidence: number): string {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    if (confidence >= 0.4) return 'text-orange-600';
    return 'text-red-600';
  }
  
  getRiskColor(risk: number): string {
    if (risk <= 0.3) return 'text-green-600';
    if (risk <= 0.6) return 'text-yellow-600';
    if (risk <= 0.8) return 'text-orange-600';
    return 'text-red-600';
  }
  
  getImpactColor(impact: string): string {
    switch (impact.toLowerCase()) {
      case 'critical': return 'text-red-800 bg-red-200';
      case 'high': return 'text-orange-800 bg-orange-200';
      case 'medium': return 'text-yellow-800 bg-yellow-200';
      case 'low': return 'text-green-800 bg-green-200';
      default: return 'text-gray-800 bg-gray-200';
    }
  }
  
  formatCurrency(value: number): string {
    if (value === undefined || value === null || isNaN(value)) {
      return '$0.00';
    }
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(value);
  }
  
  formatPercent(value: any): string {
    if (value === undefined || value === null || isNaN(value)) {
      return '0.0%';
    }
    // Convert to number if it's a string
    const numValue = typeof value === 'string' ? parseFloat(value) : value;
    if (isNaN(numValue)) {
      return '0.0%';
    }
    return `${(numValue * 100).toFixed(1)}%`;
  }
  
  formatNumber(value: any, decimals: number = 2): string {
    if (value === undefined || value === null || isNaN(value)) {
      return `0.${'0'.repeat(decimals)}`;
    }
    // Convert to number if it's a string
    const numValue = typeof value === 'string' ? parseFloat(value) : value;
    if (isNaN(numValue)) {
      return `0.${'0'.repeat(decimals)}`;
    }
    return numValue.toFixed(decimals);
  }

  formatDateTime(dateTimeString: string): string {
    if (!dateTimeString) {
      return 'N/A';
    }
    try {
      const date = new Date(dateTimeString);
      return date.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
        hour12: true
      });
    } catch (e) {
      return 'Invalid Date';
    }
  }

  getTimeRemaining(dateTimeString: string): string {
    if (!dateTimeString) {
      return 'N/A';
    }
    try {
      const targetDate = new Date(dateTimeString);
      const now = new Date();
      const diffMs = targetDate.getTime() - now.getTime();
      
      if (diffMs <= 0) {
        return 'Expired';
      }
      
      const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
      const diffMinutes = Math.floor((diffMs % (1000 * 60 * 60)) / (1000 * 60));
      
      if (diffHours > 0) {
        return `${diffHours}h ${diffMinutes}m`;
      } else {
        return `${diffMinutes}m`;
      }
    } catch (e) {
      return 'N/A';
    }
  }
  
  formatDate(dateString: string): string {
    return new Date(dateString).toLocaleString();
  }
  
  getTimeUntilExpiry(validUntil: string): string {
    const now = new Date();
    const expiry = new Date(validUntil);
    const diff = expiry.getTime() - now.getTime();
    
    if (diff <= 0) return 'Expired';
    
    const hours = Math.floor(diff / (1000 * 60 * 60));
    const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
    
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    } else {
      return `${minutes}m`;
    }
  }

  getSectorColor(sector: string): string {
    switch (sector.toLowerCase()) {
      case 'technology': return 'bg-blue-100 text-blue-800 border-blue-200';
      case 'finance': return 'bg-green-100 text-green-800 border-green-200';
      case 'healthcare': return 'bg-red-100 text-red-800 border-red-200';
      case 'retail': return 'bg-purple-100 text-purple-800 border-purple-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  }

  getSectorIcon(sector: string): string {
    switch (sector.toLowerCase()) {
      case 'technology': return '💻';
      case 'finance': return '💰';
      case 'healthcare': return '🏥';
      case 'retail': return '🛒';
      default: return '📊';
    }
  }

  formatTimestamp(timestamp: string): string {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    
    if (diffMins < 60) {
      return `${diffMins}m ago`;
    } else if (diffHours < 24) {
      return `${diffHours}h ago`;
    } else {
      return date.toLocaleDateString();
    }
  }

  toggleRAGSectorExpansion(sector: string): void {
    if (this.expandedRAGSectors.has(sector)) {
      this.expandedRAGSectors.delete(sector);
    } else {
      this.expandedRAGSectors.add(sector);
    }
  }

  isRAGSectorExpanded(sector: string): boolean {
    return this.expandedRAGSectors.has(sector);
  }

  formatMarkdown(text: string): SafeHtml {
    if (!text) return '';
    
    // Convert markdown to HTML
    let html = text
      // Headers
      .replace(/### (.*?)$/gm, '<h3 class="font-bold text-base mt-3 mb-2">$1</h3>')
      .replace(/## (.*?)$/gm, '<h2 class="font-bold text-lg mt-4 mb-2">$1</h2>')
      .replace(/# (.*?)$/gm, '<h1 class="font-bold text-xl mt-4 mb-2">$1</h1>')
      // Bold
      .replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold">$1</strong>')
      // Lists
      .replace(/^\d+\.\s+(.*?)$/gm, '<li class="ml-4">$1</li>')
      .replace(/^-\s+(.*?)$/gm, '<li class="ml-4">$1</li>')
      .replace(/^\*\s+(.*?)$/gm, '<li class="ml-4">$1</li>')
      // Line breaks
      .replace(/\n\n/g, '<br><br>')
      .replace(/\n/g, '<br>');
    
    // Wrap lists in ul tags
    html = html.replace(/(<li class="ml-4">.*?<\/li>(?:<br>)?)+/g, (match) => {
      return '<ul class="list-disc ml-4 space-y-1">' + match.replace(/<br>/g, '') + '</ul>';
    });
    
    return this.sanitizer.sanitize(1, html) || '';
  }

  // Advanced Forecasting Methods
  loadAdvancedForecasts(): void {
    this.isLoadingAdvanced = true;
    this.http.get<any>(`${this.apiUrl}/forecasting/advanced-forecasts`)
      .subscribe({
        next: (response) => {
          if (response && response.forecasts) {
            this.advancedForecasts = response.forecasts.filter((f: any) => f.status === 'success');
            console.log(`Loaded ${this.advancedForecasts.length} advanced forecasts from ${response.agents_integrated.length} agents`);
            
            // Save to database
            if (this.advancedForecasts.length > 0) {
              this.http.post(`${this.apiUrl}/forecasting/advanced-forecasts/save`, response.forecasts)
                .pipe(
                  catchError(error => {
                    console.error('Error saving advanced forecasts to database:', error);
                    return of(null);
                  })
                )
                .subscribe(saveResponse => {
                  if (saveResponse) {
                    console.log('Advanced forecasts saved to database:', saveResponse);
                  }
                });
            }
          }
          this.isLoadingAdvanced = false;
        },
        error: (error) => {
          console.error('Error loading advanced forecasts:', error);
          this.error = 'Failed to load advanced forecasts';
          this.isLoadingAdvanced = false;
        }
      });
  }
}
