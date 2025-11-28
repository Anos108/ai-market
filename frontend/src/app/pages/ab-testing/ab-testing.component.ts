import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SystemStatusService, ABTestSummary, ABTestResult } from '../../services/system-status.service';
import { Observable } from 'rxjs';

@Component({
  selector: 'app-ab-testing',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="space-y-6 max-w-7xl mx-auto">
      <!-- Page Header -->
      <div>
        <h1 class="text-3xl font-bold text-gray-900">A/B Testing</h1>
        <p class="text-gray-600">Strategy comparison and optimization</p>
      </div>

      <!-- A/B Testing Overview Cards -->
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6" *ngIf="abTestingSummary$ | async as summary">
        <div class="metric-card-enhanced">
          <div class="flex items-center justify-between">
            <div class="flex items-center">
              <div class="flex-shrink-0">
                <div class="w-12 h-12 bg-gradient-to-br from-blue-400 to-blue-600 rounded-xl flex items-center justify-center shadow-lg">
                  <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"></path>
                  </svg>
                </div>
              </div>
              <div class="ml-4">
                <p class="text-sm font-medium text-gray-600">Active Tests</p>
                <p class="text-2xl font-bold text-gray-900">{{ summary.active_experiments }}</p>
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
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                  </svg>
                </div>
              </div>
              <div class="ml-4">
                <p class="text-sm font-medium text-gray-600">Completed Tests</p>
                <p class="text-2xl font-bold text-gray-900">{{ summary.completed_experiments }}</p>
              </div>
            </div>
            <div class="text-right">
              <p class="text-xs text-gray-500">Success Rate</p>
              <p class="text-sm font-semibold text-green-600">{{ (summary.success_rate * 100) | number:'1.1-1' }}%</p>
            </div>
          </div>
        </div>

        <div class="metric-card-enhanced">
          <div class="flex items-center justify-between">
            <div class="flex items-center">
              <div class="flex-shrink-0">
                <div class="w-12 h-12 bg-gradient-to-br from-purple-400 to-purple-600 rounded-xl flex items-center justify-center shadow-lg">
                  <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path>
                  </svg>
                </div>
              </div>
              <div class="ml-4">
                <p class="text-sm font-medium text-gray-600">Conversion Rate</p>
                <p class="text-2xl font-bold text-gray-900">{{ (summary.overall_conversion_rate * 100) | number:'1.1-1' }}%</p>
              </div>
            </div>
            <div class="text-right">
              <p class="text-xs text-gray-500">Total Participants</p>
              <p class="text-sm font-semibold text-purple-600">{{ summary.total_participants | number }}</p>
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
                <p class="text-sm font-medium text-gray-600">Avg Duration</p>
                <p class="text-2xl font-bold text-gray-900">{{ summary.avg_experiment_duration | number:'1.0-0' }}d</p>
              </div>
            </div>
            <div class="text-right">
              <p class="text-xs text-gray-500">Top Variant</p>
              <p class="text-sm font-semibold text-amber-600">{{ summary.top_performing_variant }}</p>
            </div>
          </div>
        </div>
      </div>

      <!-- A/B Testing Content -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <!-- Active Tests -->
        <div class="card-enhanced">
          <div class="card-header-enhanced">
            <div class="flex items-center justify-between">
              <div>
                <h3 class="text-xl font-bold text-gray-900">Active Tests</h3>
                <p class="text-sm text-gray-600 mt-1">Currently running experiments</p>
              </div>
              <div class="flex items-center space-x-2" *ngIf="abTestingPerformance$ | async as performance">
                <div class="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                <span class="text-sm font-medium text-gray-600">{{ performance.active_experiments.length }} Running</span>
              </div>
            </div>
          </div>
          <div class="p-6">
            <div class="space-y-4" *ngIf="abTestingPerformance$ | async as performance; else loadingActive">
              <div *ngFor="let experiment of performance.active_experiments; let i = index" class="flex justify-between items-center">
                <div class="flex items-center">
                  <div class="w-8 h-8 rounded-lg flex items-center justify-center mr-3" [class]="getVariantColor(experiment.variant)">
                    <span class="text-white text-xs font-bold">{{ experiment.variant }}</span>
                  </div>
                  <span class="font-semibold text-gray-900">{{ experiment.experiment_name }}</span>
                </div>
                <span class="text-sm font-semibold" [class]="getVariantTextColor(experiment.variant)">Day {{ experiment.duration_days }}</span>
              </div>
            </div>
            <ng-template #loadingActive>
              <div class="text-center py-4">
                <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto"></div>
                <p class="text-gray-500 mt-2">Loading active experiments...</p>
              </div>
            </ng-template>
          </div>
        </div>

        <!-- Test Results -->
        <div class="card-enhanced">
          <div class="card-header-enhanced">
            <div class="flex items-center justify-between">
              <div>
                <h3 class="text-xl font-bold text-gray-900">Recent Results</h3>
                <p class="text-sm text-gray-600 mt-1">Latest test outcomes</p>
              </div>
              <div class="flex items-center space-x-2" *ngIf="abTestingPerformance$ | async as performance">
                <div class="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span class="text-sm font-medium text-gray-600">{{ getWinsCount(performance.experiments) }} Wins</span>
              </div>
            </div>
          </div>
          <div class="p-6">
            <div class="space-y-4" *ngIf="abTestingPerformance$ | async as performance; else loadingResults">
              <div *ngFor="let experiment of performance.experiments; let i = index" class="flex justify-between items-center">
                <div class="flex items-center">
                  <div class="w-8 h-8 rounded-lg flex items-center justify-center mr-3" [class]="getResultColor(experiment.performance_gain)">
                    <span class="text-white text-xs font-bold">{{ experiment.performance_gain > 0 ? '✓' : '✗' }}</span>
                  </div>
                  <span class="font-semibold text-gray-900">{{ experiment.experiment_name }}</span>
                </div>
                <span class="text-sm font-semibold" [class]="getPerformanceColor(experiment.performance_gain)">
                  {{ experiment.performance_gain > 0 ? '+' : '' }}{{ (experiment.performance_gain * 100) | number:'1.1-1' }}%
                </span>
              </div>
            </div>
            <ng-template #loadingResults>
              <div class="text-center py-4">
                <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-green-500 mx-auto"></div>
                <p class="text-gray-500 mt-2">Loading recent results...</p>
              </div>
            </ng-template>
          </div>
        </div>
      </div>
    </div>
  `,
  styles: []
})
export class AbTestingComponent implements OnInit {
  abTestingSummary$: Observable<ABTestSummary> | undefined;
  abTestingPerformance$: Observable<any> | undefined;

  constructor(private systemStatusService: SystemStatusService) {}

  ngOnInit() {
    this.abTestingSummary$ = this.systemStatusService.getABTestingSummary();
    this.abTestingPerformance$ = this.systemStatusService.getABTestingPerformance();
  }

  formatNumber(value: number): string {
    return new Intl.NumberFormat('en-US').format(value);
  }

  formatPercent(value: number): string {
    return `${(value * 100).toFixed(1)}%`;
  }

  getStatusColor(status: string): string {
    switch (status.toLowerCase()) {
      case 'running':
        return 'text-blue-600 bg-blue-100';
      case 'completed':
        return 'text-green-600 bg-green-100';
      case 'draft':
        return 'text-gray-600 bg-gray-100';
      case 'paused':
        return 'text-amber-600 bg-amber-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  }

  getVariantColor(variant: string): string {
    switch (variant.toUpperCase()) {
      case 'A':
        return 'bg-gradient-to-br from-blue-400 to-blue-600';
      case 'B':
        return 'bg-gradient-to-br from-green-400 to-green-600';
      case 'C':
        return 'bg-gradient-to-br from-purple-400 to-purple-600';
      default:
        return 'bg-gradient-to-br from-gray-400 to-gray-600';
    }
  }

  getVariantTextColor(variant: string): string {
    switch (variant.toUpperCase()) {
      case 'A':
        return 'text-blue-600';
      case 'B':
        return 'text-green-600';
      case 'C':
        return 'text-purple-600';
      default:
        return 'text-gray-600';
    }
  }

  getResultColor(performanceGain: number): string {
    if (performanceGain > 0) {
      return 'bg-gradient-to-br from-green-400 to-green-600';
    } else {
      return 'bg-gradient-to-br from-red-400 to-red-600';
    }
  }

  getPerformanceColor(performanceGain: number): string {
    if (performanceGain > 0) {
      return 'text-green-600';
    } else {
      return 'text-red-600';
    }
  }

  getWinsCount(experiments: any[]): number {
    return experiments.filter(exp => exp.performance_gain > 0).length;
  }
}