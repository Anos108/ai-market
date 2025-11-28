import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule, Router } from '@angular/router';

interface NavigationItem {
  name: string;
  href: string;
  icon: string;
  current: boolean;
}

interface NavigationSection {
  title: string;
  items: NavigationItem[];
  collapsed?: boolean;
}

@Component({
  selector: 'app-sidebar',
  standalone: true,
  imports: [CommonModule, RouterModule],
  template: `
    <div class="flex flex-col h-full">
      <!-- Logo -->
      <div class="flex items-center px-6 py-4 border-b border-gray-200">
        <div class="flex items-center">
          <div class="flex-shrink-0">
            <div class="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
              <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
              </svg>
            </div>
          </div>
          <div class="ml-3">
            <h1 class="text-lg font-semibold text-gray-900">AI Market Analysis</h1>
            <p class="text-xs text-gray-500">v4.17.0</p>
          </div>
        </div>
      </div>

      <!-- Navigation -->
      <nav class="flex-1 px-4 py-6 space-y-6 overflow-y-auto">
        <div *ngFor="let section of navigationSections" class="space-y-1">
          <!-- Section Header -->
          <div class="flex items-center justify-between">
            <div class="text-xs font-semibold text-gray-500 uppercase tracking-wider">
              {{ section.title }}
            </div>
            <button *ngIf="section.items.length > 4" 
                    (click)="toggleSection(section)"
                    class="text-gray-400 hover:text-gray-600 transition-colors">
              <svg class="w-4 h-4 transform transition-transform" 
                   [class.rotate-180]="!section.collapsed" 
                   fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
              </svg>
            </button>
          </div>
          
          <!-- Section Items -->
          <div class="space-y-1" [class.hidden]="section.collapsed">
            <a *ngFor="let item of section.items" 
               [routerLink]="item.href"
               [class]="getNavItemClass(item)"
               (click)="setCurrentItem(item)">
              <svg class="w-5 h-5 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" [attr.d]="item.icon"></path>
              </svg>
              {{ item.name }}
            </a>
          </div>
        </div>
      </nav>

      <!-- Footer -->
      <div class="px-4 py-4 border-t border-gray-200">
        <div class="flex items-center text-sm text-gray-500">
          <div class="w-2 h-2 bg-success-400 rounded-full mr-2"></div>
          System Online
        </div>
      </div>
    </div>
  `,
  styles: []
})
export class SidebarComponent implements OnInit {
  navigationSections: NavigationSection[] = [
    {
      title: 'Overview',
      items: [
        {
          name: 'Dashboard',
          href: '/dashboard',
          icon: 'M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2H5a2 2 0 00-2-2z',
          current: false
        },
        {
          name: 'System Status',
          href: '/system-status',
          icon: 'M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z',
          current: false
        },
        {
          name: 'Predictions',
          href: '/predictions',
          icon: 'M13 7h8m0 0v8m0-8l-8 8-4-4-6 6',
          current: false
        },
        {
          name: 'Agents',
          href: '/agents',
          icon: 'M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z',
          current: false
        }
      ]
    },
    {
      title: 'Trading & Portfolio',
      items: [
        {
          name: 'Forecasting Dashboard',
          href: '/forecasting-dashboard',
          icon: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z',
          current: false
        },
        {
          name: 'Ticker Discovery',
          href: '/ticker-discovery',
          icon: 'M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z',
          current: false
        },
        {
          name: 'Reports',
          href: '/reports',
          icon: 'M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z',
          current: false
        },
        {
          name: 'Symbol Management',
          href: '/symbol-management',
          icon: 'M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01',
          current: false
        },
      ]
    },
    {
      title: 'Analytics & Risk',
      items: [
        {
          name: 'Analytics',
          href: '/analytics',
          icon: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z',
          current: false
        },
        {
          name: 'Risk Analysis',
          href: '/risk-analysis',
          icon: 'M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z',
          current: false
        },
        {
          name: 'A/B Testing',
          href: '/ab-testing',
          icon: 'M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01',
          current: false
        }
      ]
    },
    {
      title: 'AI Agents',
      items: [
        {
          name: 'Agent Monitor',
          href: '/agent-monitor',
          icon: 'M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z',
          current: false
        },
        {
          name: 'Agent Router',
          href: '/agent-router',
          icon: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z',
          current: false
        },
        {
          name: 'Execution Agent',
          href: '/execution-agent',
          icon: 'M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01',
          current: false
        },
        {
          name: 'RAG Event Agent',
          href: '/rag-event-agent',
          icon: 'M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z',
          current: false
        },
        {
          name: 'RL Strategy Agent',
          href: '/rl-strategy-agent',
          icon: 'M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z',
          current: false
        },
        {
          name: 'Meta-Evaluation Agent',
          href: '/meta-evaluation-agent',
          icon: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z',
          current: false
        },
        {
          name: 'Latent Pattern Detector',
          href: '/latent-pattern-detector',
          icon: 'M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z',
          current: false
        },
        {
          name: 'Ensemble Signal Blender',
          href: '/ensemble-blender',
          icon: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z',
          current: false
        }
      ],
      collapsed: true
    },
  ];

  constructor(private router: Router) {}

  ngOnInit() {
    // Set current item based on current route
    this.updateCurrentItem();
    
    // Listen for route changes
    this.router.events.subscribe(() => {
      this.updateCurrentItem();
    });
  }

  updateCurrentItem() {
    const currentUrl = this.router.url;
    
    // Reset all items
    this.navigationSections.forEach(section => {
      section.items.forEach(item => {
        item.current = item.href === currentUrl;
      });
    });
  }

  setCurrentItem(item: NavigationItem) {
    // Reset all items
    this.navigationSections.forEach(section => {
      section.items.forEach(navItem => {
        navItem.current = false;
      });
    });
    
    // Set current item
    item.current = true;
  }

  toggleSection(section: NavigationSection) {
    section.collapsed = !section.collapsed;
  }

  getNavItemClass(item: NavigationItem): string {
    const baseClass = 'sidebar-nav-item flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors duration-200';
    
    if (item.current) {
      return `${baseClass} sidebar-nav-item-active bg-primary-100 text-primary-700`;
    } else {
      return `${baseClass} sidebar-nav-item-inactive text-gray-600 hover:bg-gray-50 hover:text-gray-900`;
    }
  }
}
