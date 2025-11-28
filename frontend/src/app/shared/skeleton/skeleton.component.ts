import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-skeleton',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div *ngIf="show" class="skeleton-container">
      <div *ngIf="type === 'card'" class="skeleton-card">
        <div class="skeleton skeleton-title"></div>
        <div class="skeleton skeleton-text"></div>
        <div class="skeleton skeleton-text" [style.width.%]="80"></div>
        <div class="skeleton skeleton-text" [style.width.%]="60"></div>
      </div>

      <div *ngIf="type === 'table'" class="skeleton-table">
        <div class="skeleton-table-header">
          <div *ngFor="let header of tableHeaders" class="skeleton skeleton-text"></div>
        </div>
        <div *ngFor="let row of tableRows" class="skeleton-table-row">
          <div *ngFor="let cell of row" class="skeleton skeleton-text"></div>
        </div>
      </div>

      <div *ngIf="type === 'metrics'" class="skeleton-metrics">
        <div *ngFor="let metric of metrics" class="skeleton-metric-card">
          <div class="skeleton skeleton-text" [style.width.%]="40"></div>
          <div class="skeleton skeleton-text" [style.width.%]="70"></div>
        </div>
      </div>

      <div *ngIf="type === 'chart'" class="skeleton-chart">
        <div class="skeleton skeleton-title" [style.width.%]="30"></div>
        <div class="skeleton skeleton-chart-area"></div>
      </div>

      <div *ngIf="type === 'list'" class="skeleton-list">
        <div *ngFor="let item of listItems" class="skeleton-list-item">
          <div class="skeleton skeleton-avatar"></div>
          <div class="skeleton-list-content">
            <div class="skeleton skeleton-text" [style.width.%]="80"></div>
            <div class="skeleton skeleton-text" [style.width.%]="60"></div>
          </div>
        </div>
      </div>

      <div *ngIf="type === 'custom'" class="skeleton-custom">
        <ng-content></ng-content>
      </div>
    </div>
  `,
  styles: [`
    .skeleton-container {
      width: 100%;
    }

    .skeleton {
      background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
      background-size: 200% 100%;
      animation: loading 1.5s infinite;
      border-radius: 0.25rem;
    }

    @keyframes loading {
      0% {
        background-position: 200% 0;
      }
      100% {
        background-position: -200% 0;
      }
    }

    .skeleton-title {
      height: 1.5rem;
      width: 60%;
      margin-bottom: 1rem;
    }

    .skeleton-text {
      height: 1rem;
      margin-bottom: 0.5rem;
    }

    .skeleton-card {
      padding: 1.5rem;
      border: 1px solid #e5e7eb;
      border-radius: 0.5rem;
      margin-bottom: 1rem;
    }

    .skeleton-table {
      border: 1px solid #e5e7eb;
      border-radius: 0.5rem;
      overflow: hidden;
    }

    .skeleton-table-header {
      display: flex;
      background-color: #f9fafb;
      padding: 0.75rem 1rem;
      border-bottom: 1px solid #e5e7eb;
    }

    .skeleton-table-header > div {
      flex: 1;
      height: 1rem;
      margin-right: 1rem;
    }

    .skeleton-table-header > div:last-child {
      margin-right: 0;
    }

    .skeleton-table-row {
      display: flex;
      padding: 0.75rem 1rem;
      border-bottom: 1px solid #e5e7eb;
    }

    .skeleton-table-row > div {
      flex: 1;
      height: 1rem;
      margin-right: 1rem;
    }

    .skeleton-table-row > div:last-child {
      margin-right: 0;
    }

    .skeleton-metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem;
    }

    .skeleton-metric-card {
      padding: 1.5rem;
      border: 1px solid #e5e7eb;
      border-radius: 0.5rem;
      text-align: center;
    }

    .skeleton-metric-card > div:first-child {
      margin-bottom: 0.5rem;
    }

    .skeleton-chart {
      padding: 1.5rem;
      border: 1px solid #e5e7eb;
      border-radius: 0.5rem;
    }

    .skeleton-chart-area {
      height: 200px;
      margin-top: 1rem;
    }

    .skeleton-list {
      space-y: 1rem;
    }

    .skeleton-list-item {
      display: flex;
      align-items: center;
      padding: 1rem;
      border: 1px solid #e5e7eb;
      border-radius: 0.5rem;
      margin-bottom: 1rem;
    }

    .skeleton-avatar {
      width: 3rem;
      height: 3rem;
      border-radius: 50%;
      margin-right: 1rem;
      flex-shrink: 0;
    }

    .skeleton-list-content {
      flex: 1;
    }

    .skeleton-list-content > div:first-child {
      margin-bottom: 0.5rem;
    }

    .skeleton-custom {
      width: 100%;
    }
  `]
})
export class SkeletonComponent {
  @Input() show: boolean = true;
  @Input() type: 'card' | 'table' | 'metrics' | 'chart' | 'list' | 'custom' = 'card';
  @Input() tableHeaders: string[] = ['Header 1', 'Header 2', 'Header 3'];
  @Input() tableRows: string[][] = [
    ['Row 1 Col 1', 'Row 1 Col 2', 'Row 1 Col 3'],
    ['Row 2 Col 1', 'Row 2 Col 2', 'Row 2 Col 3'],
    ['Row 3 Col 1', 'Row 3 Col 2', 'Row 3 Col 3']
  ];
  @Input() metrics: number[] = [1, 2, 3, 4];
  @Input() listItems: number[] = [1, 2, 3, 4, 5];
}
