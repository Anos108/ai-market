import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SystemStatusService, Prediction } from '../../services/system-status.service';

@Component({
  selector: 'app-predictions',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="space-y-6 max-w-7xl mx-auto">
      <!-- Page Header -->
      <div>
        <h1 class="text-3xl font-bold text-gray-900">Predictions</h1>
        <p class="text-gray-600">Real-time market predictions from AI agents</p>
      </div>

      <!-- Predictions Overview Cards -->
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div class="metric-card-enhanced">
          <div class="flex items-center justify-between">
            <div class="flex items-center">
              <div class="flex-shrink-0">
                <div class="w-12 h-12 bg-gradient-to-br from-green-400 to-green-600 rounded-xl flex items-center justify-center shadow-lg">
                  <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
                  </svg>
                </div>
              </div>
              <div class="ml-4">
                <p class="text-sm font-medium text-gray-600">Total Predictions</p>
                <p class="text-2xl font-bold text-gray-900">{{ predictions.length }}</p>
              </div>
            </div>
            <div class="flex items-center">
              <div class="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
            </div>
          </div>
        </div>

        <div class="metric-card-enhanced">
          <div class="flex items-center justify-between">
            <div class="flex items-center">
              <div class="flex-shrink-0">
                <div class="w-12 h-12 bg-gradient-to-br from-blue-400 to-blue-600 rounded-xl flex items-center justify-center shadow-lg">
                  <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path>
                  </svg>
                </div>
              </div>
              <div class="ml-4">
                <p class="text-sm font-medium text-gray-600">Buy Signals</p>
                <p class="text-2xl font-bold text-gray-900">{{ getBuySignalsCount() }}</p>
              </div>
            </div>
            <div class="text-right">
              <p class="text-xs text-gray-500">Positive</p>
              <p class="text-sm font-semibold text-green-600">{{ getBuyPercentage() }}%</p>
            </div>
          </div>
        </div>

        <div class="metric-card-enhanced">
          <div class="flex items-center justify-between">
            <div class="flex items-center">
              <div class="flex-shrink-0">
                <div class="w-12 h-12 bg-gradient-to-br from-red-400 to-red-600 rounded-xl flex items-center justify-center shadow-lg">
                  <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6"></path>
                  </svg>
                </div>
              </div>
              <div class="ml-4">
                <p class="text-sm font-medium text-gray-600">Sell Signals</p>
                <p class="text-2xl font-bold text-gray-900">{{ getSellSignalsCount() }}</p>
              </div>
            </div>
            <div class="text-right">
              <p class="text-xs text-gray-500">Negative</p>
              <p class="text-sm font-semibold text-red-600">{{ getSellPercentage() }}%</p>
            </div>
          </div>
        </div>

        <div class="metric-card-enhanced">
          <div class="flex items-center justify-between">
            <div class="flex items-center">
              <div class="flex-shrink-0">
                <div class="w-12 h-12 bg-gradient-to-br from-amber-400 to-amber-600 rounded-xl flex items-center justify-center shadow-lg">
                  <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                  </svg>
                </div>
              </div>
              <div class="ml-4">
                <p class="text-sm font-medium text-gray-600">Hold Signals</p>
                <p class="text-2xl font-bold text-gray-900">{{ getHoldSignalsCount() }}</p>
              </div>
            </div>
            <div class="text-right">
              <p class="text-xs text-gray-500">Neutral</p>
              <p class="text-sm font-semibold text-amber-600">{{ getHoldPercentage() }}%</p>
            </div>
          </div>
        </div>
      </div>

      <!-- Predictions Table -->
      <div class="card-enhanced">
        <div class="card-header-enhanced">
          <div class="flex items-center justify-between">
            <div>
              <h3 class="text-xl font-bold text-gray-900">Recent Predictions</h3>
              <p class="text-sm text-gray-600 mt-1">Live market predictions from AI agents</p>
            </div>
            <div class="flex items-center space-x-2">
              <div class="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span class="text-sm font-medium text-gray-600">{{ predictions.length }} Predictions</span>
            </div>
          </div>
        </div>
        <div class="overflow-x-auto">
          <table class="min-w-full">
            <thead>
              <tr class="border-b border-gray-200">
                <th class="table-header-enhanced">Agent</th>
                <th class="table-header-enhanced">Symbol</th>
                <th class="table-header-enhanced">Signal</th>
                <th class="table-header-enhanced">Confidence</th>
                <th class="table-header-enhanced">Time</th>
                <th class="table-header-enhanced">Reasoning</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-gray-100">
              <tr *ngFor="let prediction of predictions; let i = index" class="table-row-enhanced" [class.bg-gray-50]="i % 2 === 0">
                <td class="table-cell-enhanced">
                  <div class="flex items-center">
                    <div class="w-8 h-8 bg-gradient-to-br from-blue-400 to-blue-600 rounded-lg flex items-center justify-center mr-3">
                      <span class="text-white text-xs font-bold">{{ prediction.agent_name.charAt(0) }}</span>
                    </div>
                    <span class="font-semibold text-gray-900">{{ prediction.agent_name }}</span>
                  </div>
                </td>
                <td class="table-cell-enhanced">
                  <span class="font-semibold text-gray-900">{{ prediction.asset_symbol }}</span>
                </td>
                <td class="table-cell-enhanced">
                  <span [class]="getSignalClassEnhanced(prediction.signal_type)">
                    <div class="w-2 h-2 rounded-full mr-2" [class]="getSignalDotClass(prediction.signal_type)"></div>
                    {{ prediction.signal_type | titlecase }}
                  </span>
                </td>
                <td class="table-cell-enhanced">
                  <div class="flex items-center">
                    <div class="w-16 bg-gray-200 rounded-full h-2 mr-3">
                      <div class="bg-gradient-to-r from-blue-400 to-blue-600 h-2 rounded-full transition-all duration-300" [style.width.%]="prediction.confidence * 100"></div>
                    </div>
                    <span class="text-sm font-semibold text-gray-700">{{ (prediction.confidence * 100) | number:'1.0-0' }}%</span>
                  </div>
                </td>
                <td class="table-cell-enhanced text-sm text-gray-500">{{ prediction.timestamp | date:'short' }}</td>
                <td class="table-cell-enhanced text-sm text-gray-600 max-w-xs truncate">{{ prediction.reasoning }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  `,
  styles: []
})
export class PredictionsComponent implements OnInit {
  predictions: Prediction[] = [];

  constructor(private systemStatusService: SystemStatusService) {}

  ngOnInit() {
    this.loadPredictions();
  }

  loadPredictions() {
    this.systemStatusService.getPredictions(50).subscribe({
      next: (predictions) => {
        this.predictions = predictions;
      },
      error: (error) => {
        console.error('Error loading predictions:', error);
      }
    });
  }

  getSignalClass(signalType: string): string {
    const baseClass = 'status-indicator';
    
    switch (signalType.toLowerCase()) {
      case 'buy':
      case 'strong_buy':
        return `${baseClass} status-active`;
      case 'sell':
      case 'strong_sell':
        return `${baseClass} status-error`;
      case 'hold':
        return `${baseClass} status-idle`;
      default:
        return `${baseClass} status-warning`;
    }
  }

  getSignalClassEnhanced(signalType: string): string {
    return 'inline-flex items-center px-3 py-1 rounded-full text-xs font-medium';
  }

  getSignalDotClass(signalType: string): string {
    switch (signalType.toLowerCase()) {
      case 'buy':
      case 'strong_buy':
        return 'bg-green-500';
      case 'sell':
      case 'strong_sell':
        return 'bg-red-500';
      case 'hold':
        return 'bg-gray-400';
      default:
        return 'bg-yellow-500';
    }
  }

  getBuySignalsCount(): number {
    return this.predictions.filter(p => 
      p.signal_type.toLowerCase().includes('buy')
    ).length;
  }

  getSellSignalsCount(): number {
    return this.predictions.filter(p => 
      p.signal_type.toLowerCase().includes('sell')
    ).length;
  }

  getHoldSignalsCount(): number {
    return this.predictions.filter(p => 
      p.signal_type.toLowerCase() === 'hold'
    ).length;
  }

  getBuyPercentage(): number {
    if (this.predictions.length === 0) return 0;
    return Math.round((this.getBuySignalsCount() / this.predictions.length) * 100);
  }

  getSellPercentage(): number {
    if (this.predictions.length === 0) return 0;
    return Math.round((this.getSellSignalsCount() / this.predictions.length) * 100);
  }

  getHoldPercentage(): number {
    if (this.predictions.length === 0) return 0;
    return Math.round((this.getHoldSignalsCount() / this.predictions.length) * 100);
  }
}
