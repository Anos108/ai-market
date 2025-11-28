import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SystemStatusService, SystemStatus, AgentStatus } from '../../services/system-status.service';
import { SkeletonComponent } from '../../shared/skeleton/skeleton.component';

@Component({
  selector: 'app-system-status',
  standalone: true,
  imports: [CommonModule, SkeletonComponent],
  template: `
    <div class="space-y-6 max-w-7xl mx-auto">
      <!-- Page Header -->
      <div>
        <div class="flex items-center justify-between">
          <div>
            <h1 class="text-3xl font-bold text-gray-900">System Status</h1>
            <p class="text-gray-600">Monitor system health and agent performance</p>
          </div>
          <div *ngIf="isLoadingSystemStatus || isLoadingAgentsStatus" class="flex items-center text-sm text-blue-600">
            <svg class="animate-spin -ml-1 mr-2 h-4 w-4 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Loading system data...
          </div>
        </div>
      </div>

      <!-- System Overview -->
      <div *ngIf="!isLoadingSystemStatus; else systemOverviewLoading" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
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
                <p class="text-sm font-medium text-gray-600">System Status</p>
                <p class="text-2xl font-bold text-gray-900">{{ systemStatus?.is_running ? 'Online' : 'Offline' }}</p>
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
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"></path>
                  </svg>
                </div>
              </div>
              <div class="ml-4">
                <p class="text-sm font-medium text-gray-600">Active Agents</p>
                <p class="text-2xl font-bold text-gray-900">{{ systemStatus?.active_agents?.length || 0 }}</p>
              </div>
            </div>
            <div class="text-right">
              <p class="text-xs text-gray-500">Total</p>
              <p class="text-sm font-semibold text-blue-600">10</p>
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
                <p class="text-sm font-medium text-gray-600">Uptime</p>
                <p class="text-2xl font-bold text-gray-900">{{ formatUptime(systemStatus?.uptime_seconds) }}</p>
              </div>
            </div>
            <div class="text-right">
              <p class="text-xs text-gray-500">Since</p>
              <p class="text-sm font-semibold text-amber-600">Start</p>
            </div>
          </div>
        </div>

        <div class="metric-card-enhanced">
          <div class="flex items-center justify-between">
            <div class="flex items-center">
              <div class="flex-shrink-0">
                <div class="w-12 h-12 bg-gradient-to-br from-purple-400 to-purple-600 rounded-xl flex items-center justify-center shadow-lg">
                  <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
                  </svg>
                </div>
              </div>
              <div class="ml-4">
                <p class="text-sm font-medium text-gray-600">Data Quality</p>
                <p class="text-2xl font-bold text-gray-900">{{ formatPercentage(systemStatus?.data_quality_score) }}%</p>
              </div>
            </div>
            <div class="text-right">
              <div class="w-12 h-2 bg-gray-200 rounded-full">
                <div class="h-2 bg-gradient-to-r from-purple-400 to-purple-600 rounded-full" [style.width.%]="(systemStatus?.data_quality_score || 0) * 100"></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <ng-template #systemOverviewLoading>
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <app-skeleton type="metrics" [metrics]="[1,2,3,4]"></app-skeleton>
        </div>
      </ng-template>

      <!-- Agent Status Table -->
      <div class="card-enhanced">
        <div class="card-header-enhanced">
          <div class="flex items-center justify-between">
            <div>
              <h3 class="text-xl font-bold text-gray-900">Agent Status</h3>
              <p class="text-sm text-gray-600 mt-1">Monitor individual agent performance and status</p>
            </div>
            <div class="flex items-center space-x-2">
              <div class="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span class="text-sm font-medium text-gray-600">{{ agentsStatus.length }} Agents</span>
            </div>
          </div>
        </div>
        <div *ngIf="!isLoadingAgentsStatus; else agentsTableLoading" class="overflow-x-auto">
          <table class="min-w-full">
            <thead>
              <tr class="border-b border-gray-200">
                <th class="table-header-enhanced">Agent Name</th>
                <th class="table-header-enhanced">Status</th>
                <th class="table-header-enhanced">Last Prediction</th>
                <th class="table-header-enhanced">Total Predictions</th>
                <th class="table-header-enhanced">Accuracy</th>
                <th class="table-header-enhanced">Confidence</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-gray-100">
              <tr *ngFor="let agent of agentsStatus; let i = index" class="table-row-enhanced" [class.bg-gray-50]="i % 2 === 0">
                <td class="table-cell-enhanced">
                  <div class="flex items-center">
                    <div class="w-8 h-8 bg-gradient-to-br from-blue-400 to-blue-600 rounded-lg flex items-center justify-center mr-3">
                      <span class="text-white text-xs font-bold">{{ agent.agent_name.charAt(0) }}</span>
                    </div>
                    <span class="font-semibold text-gray-900">{{ agent.agent_name }}</span>
                  </div>
                </td>
                <td class="table-cell-enhanced">
                  <span [class]="getStatusClassEnhanced(agent.status)">
                    <div class="w-2 h-2 rounded-full mr-2" [class]="getStatusDotClass(agent.status)"></div>
                    {{ agent.status | titlecase }}
                  </span>
                </td>
                <td class="table-cell-enhanced text-sm text-gray-500">{{ agent.last_prediction }}</td>
                <td class="table-cell-enhanced">
                  <span class="font-semibold text-gray-900">{{ agent.total_predictions || 0 }}</span>
                </td>
                <td class="table-cell-enhanced">
                  <div class="flex items-center">
                    <span class="font-semibold text-gray-900 mr-2">{{ (agent.accuracy || 0) | number:'1.1-1' }}%</span>
                    <div class="w-12 h-1 bg-gray-200 rounded-full">
                      <div class="h-1 bg-gradient-to-r from-red-400 to-green-400 rounded-full" [style.width.%]="(agent.accuracy || 0) * 100"></div>
                    </div>
                  </div>
                </td>
                <td class="table-cell-enhanced">
                  <div class="flex items-center">
                    <div class="w-16 bg-gray-200 rounded-full h-2 mr-3">
                      <div class="bg-gradient-to-r from-blue-400 to-blue-600 h-2 rounded-full transition-all duration-300" [style.width.%]="(agent.confidence || 0) * 100"></div>
                    </div>
                    <span class="text-sm font-semibold text-gray-700">{{ ((agent.confidence || 0) * 100) | number:'1.0-0' }}%</span>
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        
        <ng-template #agentsTableLoading>
          <div class="p-6">
            <app-skeleton type="table" [tableHeaders]="['Agent Name', 'Status', 'Last Prediction', 'Total Predictions', 'Accuracy', 'Confidence']" [tableRows]="[['', '', '', '', '', ''], ['', '', '', '', '', ''], ['', '', '', '', '', ''], ['', '', '', '', '', ''], ['', '', '', '', '', '']]"></app-skeleton>
          </div>
        </ng-template>
      </div>
    </div>
  `,
  styles: []
})
export class SystemStatusComponent implements OnInit {
  systemStatus: SystemStatus | null = null;
  agentsStatus: AgentStatus[] = [];
  
  // Loading states
  isLoadingSystemStatus = true;
  isLoadingAgentsStatus = true;

  constructor(private systemStatusService: SystemStatusService) {}

  ngOnInit() {
    this.loadSystemStatus();
    this.loadAgentsStatus();
  }

  loadSystemStatus() {
    this.isLoadingSystemStatus = true;
    console.log('Loading system status...');
    this.systemStatusService.getSystemStatus().subscribe({
      next: (status) => {
        console.log('System status loaded successfully:', status);
        this.systemStatus = status;
        this.isLoadingSystemStatus = false;
        console.log('System status loading state set to false');
      },
      error: (error) => {
        console.error('Error loading system status:', error);
        this.isLoadingSystemStatus = false;
        console.log('System status loading state set to false (error)');
      }
    });
  }

  loadAgentsStatus() {
    this.isLoadingAgentsStatus = true;
    console.log('Loading agents status...');
    this.systemStatusService.getAgentsStatus().subscribe({
      next: (agents) => {
        console.log('Agents status loaded successfully:', agents);
        this.agentsStatus = agents;
        this.isLoadingAgentsStatus = false;
        console.log('Agents status loading state set to false');
      },
      error: (error) => {
        console.error('Error loading agents status:', error);
        this.isLoadingAgentsStatus = false;
        console.log('Agents status loading state set to false (error)');
      }
    });
  }

  getStatusClass(status: string): string {
    const baseClass = 'status-indicator';
    
    switch (status.toLowerCase()) {
      case 'idle':
        return `${baseClass} status-idle`;
      case 'active':
      case 'running':
        return `${baseClass} status-active`;
      case 'error':
        return `${baseClass} status-error`;
      case 'warning':
        return `${baseClass} status-warning`;
      default:
        return `${baseClass} status-idle`;
    }
  }

  getStatusClassEnhanced(status: string): string {
    return 'inline-flex items-center px-3 py-1 rounded-full text-xs font-medium';
  }

  getStatusDotClass(status: string): string {
    switch (status.toLowerCase()) {
      case 'idle':
        return 'bg-gray-400';
      case 'active':
      case 'running':
        return 'bg-green-500';
      case 'error':
        return 'bg-red-500';
      case 'warning':
        return 'bg-yellow-500';
      default:
        return 'bg-gray-400';
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
}
