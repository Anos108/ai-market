import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-loading',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div *ngIf="show" class="loading-container" [class]="containerClass">
      <div class="loading-content">
        <div class="loading-spinner" [class]="spinnerClass">
          <div class="spinner"></div>
        </div>
        <div *ngIf="message" class="loading-message" [class]="messageClass">
          {{ message }}
        </div>
        <div *ngIf="subMessage" class="loading-sub-message" [class]="subMessageClass">
          {{ subMessage }}
        </div>
      </div>
    </div>
  `,
  styles: [`
    .loading-container {
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 200px;
      width: 100%;
    }

    .loading-container.full-screen {
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(255, 255, 255, 0.9);
      z-index: 9999;
      min-height: 100vh;
    }

    .loading-container.overlay {
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(255, 255, 255, 0.8);
      z-index: 1000;
    }

    .loading-container.inline {
      min-height: 100px;
      padding: 2rem;
    }

    .loading-content {
      text-align: center;
    }

    .loading-spinner {
      margin: 0 auto 1rem;
    }

    .spinner {
      width: 40px;
      height: 40px;
      border: 4px solid #f3f4f6;
      border-top: 4px solid #3b82f6;
      border-radius: 50%;
      animation: spin 1s linear infinite;
    }

    .spinner.large {
      width: 60px;
      height: 60px;
      border-width: 6px;
    }

    .spinner.small {
      width: 24px;
      height: 24px;
      border-width: 3px;
    }

    .spinner.primary {
      border-top-color: #3b82f6;
    }

    .spinner.success {
      border-top-color: #10b981;
    }

    .spinner.warning {
      border-top-color: #f59e0b;
    }

    .spinner.danger {
      border-top-color: #ef4444;
    }

    .loading-message {
      font-size: 1.125rem;
      font-weight: 600;
      color: #374151;
      margin-bottom: 0.5rem;
    }

    .loading-sub-message {
      font-size: 0.875rem;
      color: #6b7280;
    }

    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }

    /* Pulse animation for additional loading states */
    .pulse {
      animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }

    @keyframes pulse {
      0%, 100% {
        opacity: 1;
      }
      50% {
        opacity: .5;
      }
    }

    /* Skeleton loading styles */
    .skeleton {
      background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
      background-size: 200% 100%;
      animation: loading 1.5s infinite;
    }

    @keyframes loading {
      0% {
        background-position: 200% 0;
      }
      100% {
        background-position: -200% 0;
      }
    }

    .skeleton-text {
      height: 1rem;
      border-radius: 0.25rem;
      margin-bottom: 0.5rem;
    }

    .skeleton-title {
      height: 1.5rem;
      width: 60%;
      margin-bottom: 1rem;
    }

    .skeleton-card {
      height: 120px;
      border-radius: 0.5rem;
      margin-bottom: 1rem;
    }
  `]
})
export class LoadingComponent {
  @Input() show: boolean = true;
  @Input() message: string = 'Loading...';
  @Input() subMessage: string = '';
  @Input() type: 'default' | 'full-screen' | 'overlay' | 'inline' = 'default';
  @Input() spinnerSize: 'small' | 'default' | 'large' = 'default';
  @Input() spinnerColor: 'primary' | 'success' | 'warning' | 'danger' = 'primary';

  get containerClass(): string {
    return `loading-container ${this.type}`;
  }

  get spinnerClass(): string {
    return `spinner ${this.spinnerSize} ${this.spinnerColor}`;
  }

  get messageClass(): string {
    return 'loading-message';
  }

  get subMessageClass(): string {
    return 'loading-sub-message';
  }
}
