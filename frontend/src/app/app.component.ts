import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterOutlet } from '@angular/router';
import { SidebarComponent } from './components/sidebar/sidebar.component';
import { HeaderComponent } from './components/header/header.component';
import { SystemStatusService } from './services/system-status.service';
import { ModalComponent } from './shared/modal/modal.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, RouterOutlet, SidebarComponent, HeaderComponent, ModalComponent],
  template: `
    <div class="min-h-screen bg-gray-50">
      <!-- Sidebar -->
      <div class="fixed inset-y-0 left-0 z-50 w-64 bg-white shadow-lg sidebar-responsive">
        <app-sidebar></app-sidebar>
      </div>
      
      <!-- Main content -->
      <div class="flex flex-col min-h-screen main-content-responsive">
        <div class="flex-1 p-6 lg:p-8">
        <!-- Header -->
        <app-header></app-header>
        
        <!-- Page content -->
        <main class="flex-1 overflow-auto">
          <router-outlet></router-outlet>
        </main>
        </div>
      </div>
      
      <!-- Global Modal -->
      <app-modal></app-modal>
    </div>
  `,
  styles: []
})
export class AppComponent implements OnInit {
  title = 'AI Market Analysis System';

  constructor(private systemStatusService: SystemStatusService) {}

  ngOnInit() {
    // Initialize system status monitoring
    this.systemStatusService.startPolling();
  }
}
