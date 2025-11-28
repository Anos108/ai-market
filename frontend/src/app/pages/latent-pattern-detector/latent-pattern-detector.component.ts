import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Observable } from 'rxjs';
import { SystemStatusService, LatentPatternSummary, LatentPattern, CompressionMetric, PatternInsight } from '../../services/system-status.service';

@Component({
  selector: 'app-latent-pattern-detector',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './latent-pattern-detector.component.html',
  styles: []
})
export class LatentPatternDetectorComponent implements OnInit {
  latentPatternSummary$: Observable<LatentPatternSummary> | undefined;
  latentPatterns$: Observable<LatentPattern[]> | undefined;
  compressionMetrics$: Observable<CompressionMetric[]> | undefined;
  patternInsights$: Observable<PatternInsight[]> | undefined;

  // Loading states
  isLoadingSummary = true;
  isLoadingPatterns = true;
  isLoadingMetrics = true;
  isLoadingInsights = true;

  Object = Object; // Make Object available in template

  constructor(private systemStatusService: SystemStatusService) {}

  ngOnInit() {
    this.loadLatentPatternData();
  }

  loadLatentPatternData() {
    // Load summary
    this.latentPatternSummary$ = this.systemStatusService.getLatentPatternSummary();
    this.latentPatternSummary$.subscribe({
      next: () => this.isLoadingSummary = false,
      error: () => this.isLoadingSummary = false
    });

    // Load patterns
    this.latentPatterns$ = this.systemStatusService.getLatentPatterns();
    this.latentPatterns$.subscribe({
      next: () => this.isLoadingPatterns = false,
      error: () => this.isLoadingPatterns = false
    });

    // Load compression metrics
    this.compressionMetrics$ = this.systemStatusService.getCompressionMetrics();
    this.compressionMetrics$.subscribe({
      next: () => this.isLoadingMetrics = false,
      error: () => this.isLoadingMetrics = false
    });

    // Load pattern insights
    this.patternInsights$ = this.systemStatusService.getPatternInsights();
    this.patternInsights$.subscribe({
      next: () => this.isLoadingInsights = false,
      error: () => this.isLoadingInsights = false
    });
  }

  getPatternTypeColor(patternType: string): string {
    switch (patternType.toLowerCase()) {
      case 'trend': return 'bg-green-100 text-green-800';
      case 'regime': return 'bg-blue-100 text-blue-800';
      case 'anomaly': return 'bg-red-100 text-red-800';
      case 'cyclical': return 'bg-yellow-100 text-yellow-800';
      case 'volatility': return 'bg-purple-100 text-purple-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  }

  getMethodColor(method: string): string {
    switch (method.toLowerCase()) {
      case 'pca': return 'bg-blue-100 text-blue-800';
      case 'autoencoder': return 'bg-green-100 text-green-800';
      case 'tsne': return 'bg-yellow-100 text-yellow-800';
      case 'umap': return 'bg-purple-100 text-purple-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  }

  getConfidenceColor(confidence: number): string {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  }
}
