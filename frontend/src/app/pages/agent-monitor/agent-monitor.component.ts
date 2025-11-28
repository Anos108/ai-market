import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SystemStatusService, AgentMonitorSummary, AgentPerformanceMetrics, AgentFeedback, OnlineLearningStatus } from '../../services/system-status.service';
import { Observable } from 'rxjs';

@Component({
  selector: 'app-agent-monitor',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="space-y-6 max-w-7xl mx-auto">
      <!-- Page Header -->
      <div>
        <h1 class="text-3xl font-bold text-gray-900">Agent Monitor</h1>
        <p class="text-gray-600">Agent performance tracking and online learning</p>
        <!-- Loading indicator -->
        <div *ngIf="isLoadingSummary || isLoadingMetrics || isLoadingLearning || isLoadingFeedback" class="mt-4 flex items-center justify-center">
          <div class="flex items-center space-x-3">
            <div class="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-500"></div>
            <span class="text-sm text-gray-600">Loading agent data...</span>
          </div>
        </div>
      </div>

      <!-- Agent Monitor Overview Cards -->
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <ng-container *ngIf="agentMonitorSummary$ | async as summary; else summaryLoading">
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
                <p class="text-sm font-medium text-gray-600">Healthy Agents</p>
                <p class="text-2xl font-bold text-gray-900">{{ summary.healthy_agents }}/{{ summary.total_agents }}</p>
              </div>
            </div>
            <div class="flex items-center">
              <div class="w-3 h-3 rounded-full animate-pulse" [class]="getHealthColor(summary.healthy_agents, summary.total_agents)"></div>
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
                <p class="text-sm font-medium text-gray-600">Avg Accuracy</p>
                <p class="text-2xl font-bold text-gray-900">{{ (summary.avg_accuracy * 100) | number:'1.1-1' }}%</p>
              </div>
            </div>
            <div class="text-right">
              <p class="text-xs text-gray-500">Sharpe Ratio</p>
              <p class="text-sm font-semibold text-blue-600">{{ summary.avg_sharpe_ratio | number:'1.2-2' }}</p>
            </div>
          </div>
        </div>

        <div class="metric-card-enhanced">
          <div class="flex items-center justify-between">
            <div class="flex items-center">
              <div class="flex-shrink-0">
                <div class="w-12 h-12 bg-gradient-to-br from-purple-400 to-purple-600 rounded-xl flex items-center justify-center shadow-lg">
                  <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"></path>
                  </svg>
                </div>
              </div>
              <div class="ml-4">
                <p class="text-sm font-medium text-gray-600">Online Learning</p>
                <p class="text-2xl font-bold text-gray-900">{{ summary.online_learning_enabled ? 'Active' : 'Inactive' }}</p>
              </div>
            </div>
            <div class="text-right">
              <p class="text-xs text-gray-500">Samples</p>
              <p class="text-sm font-semibold text-purple-600">{{ summary.total_feedback_samples | number }}</p>
            </div>
          </div>
        </div>

        <div class="metric-card-enhanced">
          <div class="flex items-center justify-between">
            <div class="flex items-center">
              <div class="flex-shrink-0">
                <div class="w-12 h-12 bg-gradient-to-br from-amber-400 to-amber-600 rounded-xl flex items-center justify-center shadow-lg">
                  <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z"></path>
                  </svg>
                </div>
              </div>
              <div class="ml-4">
                <p class="text-sm font-medium text-gray-600">Need Attention</p>
                <p class="text-2xl font-bold text-gray-900">{{ summary.agents_needing_attention }}</p>
              </div>
            </div>
            <div class="text-right">
              <p class="text-xs text-gray-500">Win Rate</p>
              <p class="text-sm font-semibold text-amber-600">{{ (summary.avg_win_rate * 100) | number:'1.1-1' }}%</p>
            </div>
          </div>
        </div>
        </ng-container>
        
        <!-- Loading template for summary cards -->
        <ng-template #summaryLoading>
          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div *ngFor="let i of [1,2,3,4]" class="metric-card-enhanced animate-pulse">
              <div class="flex items-center justify-between">
                <div class="flex items-center">
                  <div class="w-12 h-12 bg-gray-300 rounded-xl"></div>
                  <div class="ml-4">
                    <div class="h-4 bg-gray-300 rounded w-20 mb-2"></div>
                    <div class="h-6 bg-gray-300 rounded w-12"></div>
                  </div>
                </div>
                <div class="w-3 h-3 bg-gray-300 rounded-full"></div>
              </div>
            </div>
          </div>
        </ng-template>
      </div>

      <!-- Agent Performance Table -->
      <div class="metric-card-enhanced">
        <ng-container *ngIf="agentPerformanceMetrics$ | async as metrics; else metricsLoading">
        <div class="px-6 py-4 border-b border-gray-200">
          <h3 class="text-lg font-medium text-gray-900">Agent Performance Metrics</h3>
          <p class="text-sm text-gray-600">Real-time performance tracking for all agents</p>
        </div>
        <div class="overflow-x-auto">
          <table class="min-w-full divide-y divide-gray-200">
            <thead class="bg-gray-50">
              <tr>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Agent</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Accuracy</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Sharpe Ratio</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Win Rate</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Health Score</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Trend</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Predictions</th>
              </tr>
            </thead>
            <tbody class="bg-white divide-y divide-gray-200">
              <tr *ngFor="let metric of metrics" class="hover:bg-gray-50">
                <td class="px-6 py-4 whitespace-nowrap">
                  <div class="flex items-center">
                    <div class="flex-shrink-0 h-8 w-8">
                      <div class="h-8 w-8 rounded-full bg-gradient-to-r from-blue-400 to-purple-500 flex items-center justify-center">
                        <span class="text-xs font-medium text-white">{{ metric.agent_name.charAt(0) }}</span>
                      </div>
                    </div>
                    <div class="ml-4">
                      <div class="text-sm font-medium text-gray-900">{{ metric.agent_name }}</div>
                      <div class="text-sm text-gray-500">Last: {{ metric.last_prediction_time | date:'short' }}</div>
                    </div>
                  </div>
                </td>
                <td class="px-6 py-4 whitespace-nowrap">
                  <div class="text-sm font-medium text-gray-900">{{ (metric.accuracy * 100) | number:'1.1-1' }}%</div>
                  <div class="text-sm text-gray-500">{{ metric.correct_predictions }}/{{ metric.total_predictions }}</div>
                </td>
                <td class="px-6 py-4 whitespace-nowrap">
                  <div class="text-sm font-medium text-gray-900">{{ metric.sharpe_ratio | number:'1.2-2' }}</div>
                  <div class="text-sm text-gray-500">Conf: {{ (metric.avg_confidence * 100) | number:'1.1-1' }}%</div>
                </td>
                <td class="px-6 py-4 whitespace-nowrap">
                  <div class="text-sm font-medium text-gray-900">{{ (metric.win_rate * 100) | number:'1.1-1' }}%</div>
                </td>
                <td class="px-6 py-4 whitespace-nowrap">
                  <div class="flex items-center">
                    <div class="text-sm font-medium text-gray-900">{{ metric.health_score | number:'1.1-1' }}</div>
                    <div class="ml-2 w-16 bg-gray-200 rounded-full h-2">
                      <div class="h-2 rounded-full" [class]="getHealthBarColor(metric.health_score)" [style.width.%]="metric.health_score"></div>
                    </div>
                  </div>
                </td>
                <td class="px-6 py-4 whitespace-nowrap">
                  <span class="inline-flex px-2 py-1 text-xs font-semibold rounded-full" [class]="getTrendColor(metric.performance_trend)">
                    {{ metric.performance_trend }}
                  </span>
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  {{ metric.total_predictions | number }}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        </ng-container>
        
        <!-- Loading template for performance table -->
        <ng-template #metricsLoading>
          <div class="px-6 py-4 border-b border-gray-200">
            <h3 class="text-lg font-medium text-gray-900">Agent Performance Metrics</h3>
            <p class="text-sm text-gray-600">Real-time performance tracking for all agents</p>
          </div>
          <div class="p-6 animate-pulse">
            <div class="space-y-4">
              <div *ngFor="let i of [1,2,3,4,5]" class="flex items-center space-x-4">
                <div class="w-8 h-8 bg-gray-300 rounded-full"></div>
                <div class="flex-1 space-y-2">
                  <div class="h-4 bg-gray-300 rounded w-24"></div>
                  <div class="h-3 bg-gray-300 rounded w-16"></div>
                </div>
                <div class="w-16 h-4 bg-gray-300 rounded"></div>
                <div class="w-16 h-4 bg-gray-300 rounded"></div>
                <div class="w-16 h-4 bg-gray-300 rounded"></div>
                <div class="w-16 h-4 bg-gray-300 rounded"></div>
                <div class="w-16 h-4 bg-gray-300 rounded"></div>
              </div>
            </div>
          </div>
        </ng-template>
      </div>

      <!-- Online Learning Status -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div class="metric-card-enhanced">
          <ng-container *ngIf="onlineLearningStatus$ | async as learningStatus; else learningLoading">
          <div class="px-6 py-4 border-b border-gray-200">
            <h3 class="text-lg font-medium text-gray-900">Online Learning Status</h3>
            <p class="text-sm text-gray-600">Real-time model training and adaptation</p>
          </div>
          <div class="p-6 space-y-4">
            <div *ngFor="let status of learningStatus" class="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
              <div class="flex items-center">
                <div class="flex-shrink-0">
                  <div class="w-8 h-8 rounded-full bg-gradient-to-r from-green-400 to-blue-500 flex items-center justify-center">
                    <span class="text-xs font-medium text-white">{{ status.agent_name.charAt(0) }}</span>
                  </div>
                </div>
                <div class="ml-3">
                  <div class="text-sm font-medium text-gray-900">{{ status.agent_name }}</div>
                  <div class="text-sm text-gray-500">{{ status.model_type }}</div>
                </div>
              </div>
              <div class="text-right">
                <div class="text-sm font-medium text-gray-900">{{ (status.model_accuracy * 100) | number:'1.1-1' }}%</div>
                <div class="text-sm text-gray-500">{{ status.training_samples | number }} samples</div>
                <div class="flex items-center mt-1">
                  <div class="w-2 h-2 rounded-full mr-2" [class]="status.is_training ? 'bg-green-500 animate-pulse' : 'bg-gray-400'"></div>
                  <span class="text-xs text-gray-500">{{ status.is_training ? 'Training' : 'Idle' }}</span>
                </div>
              </div>
            </div>
          </div>
          </ng-container>
          
          <!-- Loading template for online learning -->
          <ng-template #learningLoading>
            <div class="px-6 py-4 border-b border-gray-200">
              <h3 class="text-lg font-medium text-gray-900">Online Learning Status</h3>
              <p class="text-sm text-gray-600">Real-time model training and adaptation</p>
            </div>
            <div class="p-6 animate-pulse">
              <div class="space-y-4">
                <div *ngFor="let i of [1,2,3]" class="flex items-center justify-between p-4 bg-gray-100 rounded-lg">
                  <div class="flex items-center">
                    <div class="w-8 h-8 bg-gray-300 rounded-full"></div>
                    <div class="ml-3">
                      <div class="h-4 bg-gray-300 rounded w-24 mb-1"></div>
                      <div class="h-3 bg-gray-300 rounded w-16"></div>
                    </div>
                  </div>
                  <div class="text-right">
                    <div class="h-4 bg-gray-300 rounded w-12 mb-1"></div>
                    <div class="h-3 bg-gray-300 rounded w-16"></div>
                  </div>
                </div>
              </div>
            </div>
          </ng-template>
        </div>

        <div class="metric-card-enhanced">
          <ng-container *ngIf="agentFeedback$ | async as feedback; else feedbackLoading">
          <div class="px-6 py-4 border-b border-gray-200">
            <h3 class="text-lg font-medium text-gray-900">Recent Feedback</h3>
            <p class="text-sm text-gray-600">Latest prediction feedback and outcomes</p>
          </div>
          <div class="p-6 space-y-3 max-h-96 overflow-y-auto">
            <div *ngFor="let item of feedback.slice(0, 10)" class="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <div class="flex items-center">
                <div class="flex-shrink-0">
                  <div class="w-6 h-6 rounded-full bg-gradient-to-r from-blue-400 to-purple-500 flex items-center justify-center">
                    <span class="text-xs font-medium text-white">{{ item.agent_name.charAt(0) }}</span>
                  </div>
                </div>
                <div class="ml-3">
                  <div class="text-sm font-medium text-gray-900">{{ item.agent_name }}</div>
                  <div class="text-sm text-gray-500">{{ item.predicted_signal }} → {{ item.actual_outcome }}</div>
                </div>
              </div>
              <div class="text-right">
                <div class="text-sm font-medium" [class]="getFeedbackColor(item.feedback_score)">
                  {{ item.feedback_score > 0 ? '+' : '' }}{{ item.feedback_score | number:'1.2-2' }}
                </div>
                <div class="text-sm text-gray-500">{{ item.timestamp | date:'short' }}</div>
              </div>
            </div>
          </div>
          </ng-container>
          
          <!-- Loading template for recent feedback -->
          <ng-template #feedbackLoading>
            <div class="px-6 py-4 border-b border-gray-200">
              <h3 class="text-lg font-medium text-gray-900">Recent Feedback</h3>
              <p class="text-sm text-gray-600">Latest prediction feedback and outcomes</p>
            </div>
            <div class="p-6 animate-pulse">
              <div class="space-y-3">
                <div *ngFor="let i of [1,2,3,4,5]" class="flex items-center justify-between p-3 bg-gray-100 rounded-lg">
                  <div class="flex items-center">
                    <div class="w-6 h-6 bg-gray-300 rounded-full"></div>
                    <div class="ml-3">
                      <div class="h-4 bg-gray-300 rounded w-20 mb-1"></div>
                      <div class="h-3 bg-gray-300 rounded w-24"></div>
                    </div>
                  </div>
                  <div class="text-right">
                    <div class="h-4 bg-gray-300 rounded w-12 mb-1"></div>
                    <div class="h-3 bg-gray-300 rounded w-16"></div>
                  </div>
                </div>
              </div>
            </div>
          </ng-template>
        </div>
      </div>
    </div>
  `,
  styles: []
})
export class AgentMonitorComponent implements OnInit {
  agentMonitorSummary$: Observable<AgentMonitorSummary> | undefined;
  agentPerformanceMetrics$: Observable<AgentPerformanceMetrics[]> | undefined;
  agentFeedback$: Observable<AgentFeedback[]> | undefined;
  onlineLearningStatus$: Observable<OnlineLearningStatus[]> | undefined;
  
  // Loading states
  isLoadingSummary = true;
  isLoadingMetrics = true;
  isLoadingLearning = true;
  isLoadingFeedback = true;

  constructor(private systemStatusService: SystemStatusService) {}

  ngOnInit() {
    this.loadAgentMonitorData();
  }

  loadAgentMonitorData() {
    // Load summary data
    this.isLoadingSummary = true;
    this.agentMonitorSummary$ = this.systemStatusService.getAgentMonitorSummary();
    this.agentMonitorSummary$.subscribe({
      next: () => this.isLoadingSummary = false,
      error: () => this.isLoadingSummary = false
    });

    // Load performance metrics
    this.isLoadingMetrics = true;
    this.agentPerformanceMetrics$ = this.systemStatusService.getAgentPerformanceMetrics();
    this.agentPerformanceMetrics$.subscribe({
      next: () => this.isLoadingMetrics = false,
      error: () => this.isLoadingMetrics = false
    });

    // Load online learning status
    this.isLoadingLearning = true;
    this.onlineLearningStatus$ = this.systemStatusService.getOnlineLearningStatus();
    this.onlineLearningStatus$.subscribe({
      next: () => this.isLoadingLearning = false,
      error: () => this.isLoadingLearning = false
    });

    // Load feedback data
    this.isLoadingFeedback = true;
    this.agentFeedback$ = this.systemStatusService.getAgentFeedback();
    this.agentFeedback$.subscribe({
      next: () => this.isLoadingFeedback = false,
      error: () => this.isLoadingFeedback = false
    });
  }

  getHealthColor(healthy: number, total: number): string {
    const ratio = healthy / total;
    if (ratio >= 0.8) return 'bg-green-500';
    if (ratio >= 0.6) return 'bg-yellow-500';
    return 'bg-red-500';
  }

  getHealthBarColor(score: number): string {
    if (score >= 80) return 'bg-green-500';
    if (score >= 60) return 'bg-yellow-500';
    return 'bg-red-500';
  }

  getTrendColor(trend: string): string {
    switch (trend.toLowerCase()) {
      case 'improving':
        return 'bg-green-100 text-green-800';
      case 'stable':
        return 'bg-blue-100 text-blue-800';
      case 'declining':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  }

  getFeedbackColor(score: number): string {
    if (score > 0.5) return 'text-green-600';
    if (score > 0) return 'text-blue-600';
    if (score > -0.5) return 'text-yellow-600';
    return 'text-red-600';
  }

  formatNumber(value: number): string {
    return new Intl.NumberFormat('en-US').format(value);
  }

  formatPercent(value: number): string {
    return `${(value * 100).toFixed(1)}%`;
  }
}
