import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ModalService, ModalConfig } from './modal.service';
import { Subscription } from 'rxjs';
import { animate, style, transition, trigger } from '@angular/animations';

@Component({
  selector: 'app-modal',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div *ngIf="config" 
         class="fixed inset-0 z-50 overflow-y-auto"
         role="dialog"
         aria-modal="true">
      
      <!-- Backdrop -->
      <div class="fixed inset-0 bg-black bg-opacity-50 transition-opacity"
           [@fadeIn]></div>
      
      <!-- Modal Container -->
      <div class="flex min-h-full items-center justify-center p-4 text-center sm:p-0">
        
        <!-- Modal Panel -->
        <div class="relative transform overflow-hidden rounded-lg bg-white text-left shadow-xl transition-all sm:my-8 sm:w-full sm:max-w-lg"
             [@slideIn]>
          
          <!-- Header -->
          <div class="bg-white px-4 pb-4 pt-5 sm:p-6 sm:pb-4">
            <div class="flex items-center">
              
              <!-- Icon -->
              <div *ngIf="config.type" class="flex-shrink-0 mr-4">
                <div class="w-10 h-10 rounded-full flex items-center justify-center"
                     [class]="getIconClass(config.type)">
                  <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path *ngIf="config.type === 'success'" stroke-linecap="round" stroke-linejoin="round" 
                          stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    <path *ngIf="config.type === 'error'" stroke-linecap="round" stroke-linejoin="round" 
                          stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.5 0L4.268 19.5c-.77.833.19 2.5 1.732 2.5z"></path>
                    <path *ngIf="config.type === 'warning'" stroke-linecap="round" stroke-linejoin="round" 
                          stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.5 0L4.268 19.5c-.77.833.19 2.5 1.732 2.5z"></path>
                    <path *ngIf="config.type === 'info' || config.type === 'confirm'" stroke-linecap="round" 
                          stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                  </svg>
                </div>
              </div>
              
              <!-- Content -->
              <div class="flex-1">
                <h3 class="text-lg font-medium text-gray-900" id="modal-title">
                  {{ config.title }}
                </h3>
                <div class="mt-2">
                  <p class="text-sm text-gray-500">
                    {{ config.message }}
                  </p>
                </div>
              </div>
              
              <!-- Close Button -->
              <div class="ml-3">
                <button (click)="onCancel()"
                        class="bg-white rounded-md text-gray-400 hover:text-gray-600 focus:outline-none">
                  <span class="sr-only">Close</span>
                  <svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                  </svg>
                </button>
              </div>
            </div>
          </div>
          
          <!-- Footer -->
          <div class="bg-gray-50 px-4 py-3 sm:flex sm:flex-row-reverse sm:px-6">
            
            <!-- Confirm Button -->
            <button *ngIf="config" 
                    (click)="onConfirm()"
                    type="button"
                    class="inline-flex w-full justify-center rounded-md bg-blue-600 px-3 py-2 text-sm font-semibold text-white shadow-sm hover:bg-blue-500 sm:ml-3 sm:w-auto"
                    [class]="getConfirmButtonClass(config.type)">
              {{ config.confirmText || 'OK' }}
            </button>
            
            <!-- Cancel Button -->
            <button *ngIf="config && config.showCancel"
                    (click)="onCancel()"
                    type="button"
                    class="mt-3 inline-flex w-full justify-center rounded-md bg-white px-3 py-2 text-sm font-semibold text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 hover:bg-gray-50 sm:mt-0 sm:w-auto">
              {{ (config && config.cancelText) || 'Cancel' }}
            </button>
          </div>
        </div>
      </div>
    </div>
  `,
  animations: [
    trigger('fadeIn', [
      transition(':enter', [
        style({ opacity: 0 }),
        animate('200ms ease-in', style({ opacity: 1 }))
      ])
    ]),
    trigger('slideIn', [
      transition(':enter', [
        style({ transform: 'translateY(-50px)', opacity: 0 }),
        animate('200ms ease-out', style({ transform: 'translateY(0)', opacity: 1 }))
      ]),
      transition(':leave', [
        style({ transform: 'translateY(0)', opacity: 1 }),
        animate('150ms ease-in', style({ transform: 'translateY(-50px)', opacity: 0 }))
      ])
    ])
  ]
})
export class ModalComponent implements OnInit, OnDestroy {
  config: ModalConfig | null = null;
  private subscription: Subscription = new Subscription();

  constructor(private modalService: ModalService) {}

  ngOnInit() {
    this.subscription.add(
      this.modalService.modal$.subscribe(config => {
        this.config = config;
      })
    );
  }

  ngOnDestroy() {
    this.subscription.unsubscribe();
  }

  onConfirm() {
    if (this.config?.onConfirm) {
      this.config.onConfirm();
    } else {
      this.modalService.hide();
    }
  }

  onCancel() {
    if (this.config?.onCancel) {
      this.config.onCancel();
    } else {
      this.modalService.hide();
    }
  }

  getIconClass(type: string): string {
    switch (type) {
      case 'success':
        return 'bg-green-100 text-green-600';
      case 'error':
        return 'bg-red-100 text-red-600';
      case 'warning':
        return 'bg-yellow-100 text-yellow-600';
      case 'confirm':
      case 'info':
      default:
        return 'bg-blue-100 text-blue-600';
    }
  }

  getConfirmButtonClass(type: string): string {
    const baseClass = 'inline-flex w-full justify-center rounded-md px-3 py-2 text-sm font-semibold text-white shadow-sm sm:ml-3 sm:w-auto';
    
    switch (type) {
      case 'success':
        return `${baseClass} bg-green-600 hover:bg-green-500`;
      case 'error':
        return `${baseClass} bg-red-600 hover:bg-red-500`;
      case 'warning':
        return `${baseClass} bg-yellow-600 hover:bg-yellow-500`;
      case 'confirm':
      case 'info':
      default:
        return `${baseClass} bg-blue-600 hover:bg-blue-500`;
    }
  }
}
