import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { SystemStatusService } from '../../services/system-status.service';

@Component({
  selector: 'app-header',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <header class="bg-white shadow-sm border-b border-gray-200">
      <div class="px-6 py-4">
        <div class="flex items-center justify-between">
          <!-- Page Title -->
          <div>
            <h1 class="text-2xl font-semibold text-gray-900">{{ pageTitle }}</h1>
            <p class="text-sm text-gray-500">{{ pageSubtitle }}</p>
          </div>

          <!-- System Status & Actions -->
          <div class="flex items-center space-x-4">
            <!-- System Status Indicator -->
            <div class="flex items-center space-x-2">
              <div [class]="getStatusIndicatorClass()"></div>
              <span class="text-sm font-medium text-gray-700">{{ systemStatus }}</span>
            </div>

            <!-- Auto-refresh Toggle -->
            <div class="flex items-center space-x-2">
              <label class="flex items-center">
                <input type="checkbox" 
                       [(ngModel)]="autoRefresh" 
                       (change)="toggleAutoRefresh()"
                       class="rounded border-gray-300 text-primary-600 shadow-sm focus:border-primary-300 focus:ring focus:ring-primary-200 focus:ring-opacity-50">
                <span class="ml-2 text-sm text-gray-700">Auto-refresh</span>
              </label>
            </div>

            <!-- Last Updated -->
            <div class="text-sm text-gray-500">
              Last updated: {{ lastUpdated | date:'short' }}
            </div>

            <!-- Refresh Button -->
            <button (click)="refreshData()" 
                    [disabled]="isLoading"
                    class="btn btn-secondary flex items-center space-x-2">
              <svg *ngIf="!isLoading" class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
              </svg>
              <div *ngIf="isLoading" class="loading-spinner w-4 h-4"></div>
              <span>Refresh</span>
            </button>
          </div>
        </div>
      </div>
    </header>
  `,
  styles: []
})
export class HeaderComponent implements OnInit {
  pageTitle = 'Dashboard';
  pageSubtitle = 'AI Market Analysis System Overview';
  systemStatus = 'Online';
  autoRefresh = true;
  lastUpdated = new Date();
  isLoading = false;

  private refreshInterval: any;

  constructor(private systemStatusService: SystemStatusService) {}

  ngOnInit() {
    // Subscribe to system status updates
    this.systemStatusService.getSystemStatus().subscribe(status => {
      this.systemStatus = status.is_running ? 'Online' : 'Offline';
      this.lastUpdated = new Date();
    });

    // Start auto-refresh if enabled
    if (this.autoRefresh) {
      this.startAutoRefresh();
    }
  }

  ngOnDestroy() {
    if (this.refreshInterval) {
      clearInterval(this.refreshInterval);
    }
  }

  toggleAutoRefresh() {
    if (this.autoRefresh) {
      this.startAutoRefresh();
    } else {
      this.stopAutoRefresh();
    }
  }

  startAutoRefresh() {
    this.refreshInterval = setInterval(() => {
      this.refreshData();
    }, 30000); // Refresh every 30 seconds
  }

  stopAutoRefresh() {
    if (this.refreshInterval) {
      clearInterval(this.refreshInterval);
      this.refreshInterval = null;
    }
  }

  refreshData() {
    this.isLoading = true;
    
    // Trigger system status refresh
    this.systemStatusService.refreshSystemStatus().subscribe({
      next: () => {
        this.isLoading = false;
        this.lastUpdated = new Date();
      },
      error: (error) => {
        console.error('Error refreshing data:', error);
        this.isLoading = false;
      }
    });
  }

  getStatusIndicatorClass(): string {
    const baseClass = 'w-3 h-3 rounded-full';
    
    switch (this.systemStatus) {
      case 'Online':
        return `${baseClass} bg-success-400`;
      case 'Offline':
        return `${baseClass} bg-danger-400`;
      case 'Warning':
        return `${baseClass} bg-warning-400`;
      default:
        return `${baseClass} bg-gray-400`;
    }
  }
}
