import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SystemStatusService, AgentStatus } from '../../services/system-status.service';

@Component({
  selector: 'app-agents',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="space-y-6 max-w-7xl mx-auto">
      <!-- Page Header -->
      <div>
        <h1 class="text-3xl font-bold text-gray-900">Agents</h1>
        <p class="text-gray-600">AI agent management and monitoring</p>
      </div>

      <!-- Agent Overview Cards -->
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
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
                <p class="text-sm font-medium text-gray-600">Total Agents</p>
                <p class="text-2xl font-bold text-gray-900">{{ agents.length }}</p>
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
                <p class="text-sm font-medium text-gray-600">Active Agents</p>
                <p class="text-2xl font-bold text-gray-900">{{ getActiveAgentsCount() }}</p>
              </div>
            </div>
            <div class="text-right">
              <p class="text-xs text-gray-500">Running</p>
              <p class="text-sm font-semibold text-green-600">{{ getActivePercentage() }}%</p>
            </div>
          </div>
        </div>

        <div class="metric-card-enhanced">
          <div class="flex items-center justify-between">
            <div class="flex items-center">
              <div class="flex-shrink-0">
                <div class="w-12 h-12 bg-gradient-to-br from-gray-400 to-gray-600 rounded-xl flex items-center justify-center shadow-lg">
                  <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4"></path>
                  </svg>
                </div>
              </div>
              <div class="ml-4">
                <p class="text-sm font-medium text-gray-600">Idle Agents</p>
                <p class="text-2xl font-bold text-gray-900">{{ getIdleAgentsCount() }}</p>
              </div>
            </div>
            <div class="text-right">
              <p class="text-xs text-gray-500">Waiting</p>
              <p class="text-sm font-semibold text-gray-600">{{ getIdlePercentage() }}%</p>
            </div>
          </div>
        </div>

        <div class="metric-card-enhanced">
          <div class="flex items-center justify-between">
            <div class="flex items-center">
              <div class="flex-shrink-0">
                <div class="w-12 h-12 bg-gradient-to-br from-purple-400 to-purple-600 rounded-xl flex items-center justify-center shadow-lg">
                  <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
                  </svg>
                </div>
              </div>
              <div class="ml-4">
                <p class="text-sm font-medium text-gray-600">Total Predictions</p>
                <p class="text-2xl font-bold text-gray-900">{{ getTotalPredictions() }}</p>
              </div>
            </div>
            <div class="text-right">
              <p class="text-xs text-gray-500">All Time</p>
              <p class="text-sm font-semibold text-purple-600">Live</p>
            </div>
          </div>
        </div>
      </div>

      <!-- Agents Grid -->
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <div *ngFor="let agent of agents" class="card-enhanced">
          <div class="p-6">
            <div class="flex items-center justify-between mb-4">
              <div class="flex items-center">
                <div class="w-12 h-12 bg-gradient-to-br from-blue-400 to-blue-600 rounded-xl flex items-center justify-center shadow-lg mr-4">
                  <span class="text-white text-lg font-bold">{{ agent.agent_name.charAt(0) }}</span>
                </div>
                <div>
                  <h3 class="text-lg font-bold text-gray-900">{{ agent.agent_name }}</h3>
                  <p class="text-sm text-gray-600">AI Agent</p>
                </div>
              </div>
              <span [class]="getStatusClassEnhanced(agent.status)">
                <div class="w-2 h-2 rounded-full mr-2" [class]="getStatusDotClass(agent.status)"></div>
                {{ agent.status | titlecase }}
              </span>
            </div>
            
            <div class="space-y-3">
              <div class="flex justify-between items-center">
                <span class="text-sm text-gray-600">Total Predictions</span>
                <span class="font-semibold text-gray-900">{{ agent.total_predictions || 0 }}</span>
              </div>
              <div class="flex justify-between items-center">
                <span class="text-sm text-gray-600">Accuracy</span>
                <span class="font-semibold text-gray-900">{{ (agent.accuracy || 0) | number:'1.1-1' }}%</span>
              </div>
              <div class="flex justify-between items-center">
                <span class="text-sm text-gray-600">Confidence</span>
                <div class="flex items-center">
                  <div class="w-16 bg-gray-200 rounded-full h-2 mr-2">
                    <div class="bg-gradient-to-r from-blue-400 to-blue-600 h-2 rounded-full" [style.width.%]="(agent.confidence || 0) * 100"></div>
                  </div>
                  <span class="text-sm font-semibold text-gray-700">{{ ((agent.confidence || 0) * 100) | number:'1.0-0' }}%</span>
                </div>
              </div>
              <div class="flex justify-between items-center">
                <span class="text-sm text-gray-600">Last Prediction</span>
                <span class="text-sm text-gray-500">{{ agent.last_prediction }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `,
  styles: []
})
export class AgentsComponent implements OnInit {
  agents: AgentStatus[] = [];

  constructor(private systemStatusService: SystemStatusService) {}

  ngOnInit() {
    this.loadAgents();
  }

  loadAgents() {
    this.systemStatusService.getAgentsStatus().subscribe({
      next: (agents) => {
        this.agents = agents;
      },
      error: (error) => {
        console.error('Error loading agents:', error);
      }
    });
  }

  getActiveAgentsCount(): number {
    return this.agents.filter(agent => 
      agent.status.toLowerCase() === 'active' || agent.status.toLowerCase() === 'running'
    ).length;
  }

  getIdleAgentsCount(): number {
    return this.agents.filter(agent => 
      agent.status.toLowerCase() === 'idle'
    ).length;
  }

  getActivePercentage(): number {
    if (this.agents.length === 0) return 0;
    return Math.round((this.getActiveAgentsCount() / this.agents.length) * 100);
  }

  getIdlePercentage(): number {
    if (this.agents.length === 0) return 0;
    return Math.round((this.getIdleAgentsCount() / this.agents.length) * 100);
  }

  getTotalPredictions(): number {
    return this.agents.reduce((total, agent) => total + (agent.total_predictions || 0), 0);
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
}
