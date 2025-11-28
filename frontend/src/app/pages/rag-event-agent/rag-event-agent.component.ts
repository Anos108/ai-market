import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SystemStatusService, RAGEventAgentSummary, NewsDocument, RAGAnalysis } from '../../services/system-status.service';
import { Observable } from 'rxjs';

@Component({
  selector: 'app-rag-event-agent',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './rag-event-agent.component.html',
  styles: []
})
export class RAGEventAgentComponent implements OnInit {
  ragEventAgentSummary$: Observable<RAGEventAgentSummary> | undefined;
  ragDocuments$: Observable<NewsDocument[]> | undefined;
  ragAnalysis$: Observable<RAGAnalysis> | undefined;
  ragPerformance$: Observable<any> | undefined;

  // Sector-specific observables
  technologyAnalysis$: Observable<RAGAnalysis> | undefined;
  financeAnalysis$: Observable<RAGAnalysis> | undefined;
  healthcareAnalysis$: Observable<RAGAnalysis> | undefined;
  retailAnalysis$: Observable<RAGAnalysis> | undefined;

  // Loading states
  isLoadingSummary = true;
  isLoadingDocuments = true;
  isLoadingAnalysis = true;
  isLoadingPerformance = true;
  
  // Sector loading states
  isLoadingTechnology = true;
  isLoadingFinance = true;
  isLoadingHealthcare = true;
  isLoadingRetail = true;
  isRefreshingAll = false;
  
  // Individual sector refresh states
  isRefreshingTechnology = false;
  isRefreshingFinance = false;
  isRefreshingHealthcare = false;
  isRefreshingRetail = false;

  // Make Object available in template
  Object = Object;

  constructor(private systemStatusService: SystemStatusService) {}

  ngOnInit() {
    this.loadRAGEventAgentData();
    this.loadSectorAnalyses();
  }

  loadRAGEventAgentData() {
    // Load RAG Event Agent Summary
    this.isLoadingSummary = true;
    this.ragEventAgentSummary$ = this.systemStatusService.getRAGEventAgentSummary();
    this.ragEventAgentSummary$.subscribe({
      next: () => {
        this.isLoadingSummary = false;
      },
      error: (error) => {
        console.error('Error loading RAG event agent summary:', error);
        this.isLoadingSummary = false;
      }
    });

    // Load RAG Documents
    this.isLoadingDocuments = true;
    this.ragDocuments$ = this.systemStatusService.getRAGDocuments();
    this.ragDocuments$.subscribe({
      next: () => {
        this.isLoadingDocuments = false;
      },
      error: (error) => {
        console.error('Error loading RAG documents:', error);
        this.isLoadingDocuments = false;
      }
    });

    // Load RAG Analysis
    this.isLoadingAnalysis = true;
    this.ragAnalysis$ = this.systemStatusService.getRAGAnalysis();
    this.ragAnalysis$.subscribe({
      next: () => {
        this.isLoadingAnalysis = false;
      },
      error: (error) => {
        console.error('Error loading RAG analysis:', error);
        this.isLoadingAnalysis = false;
      }
    });

    // Load RAG Performance
    this.isLoadingPerformance = true;
    this.ragPerformance$ = this.systemStatusService.getRAGPerformance();
    this.ragPerformance$.subscribe({
      next: () => {
        this.isLoadingPerformance = false;
      },
      error: (error) => {
        console.error('Error loading RAG performance:', error);
        this.isLoadingPerformance = false;
      }
    });
  }

  loadSectorAnalyses() {
    // Load Technology Analysis
    this.isLoadingTechnology = true;
    this.technologyAnalysis$ = this.systemStatusService.getSectorAnalysis('technology');
    this.technologyAnalysis$.subscribe({
      next: () => {
        this.isLoadingTechnology = false;
      },
      error: (error) => {
        console.error('Error loading technology analysis:', error);
        this.isLoadingTechnology = false;
      }
    });

    // Load Finance Analysis
    this.isLoadingFinance = true;
    this.financeAnalysis$ = this.systemStatusService.getSectorAnalysis('finance');
    this.financeAnalysis$.subscribe({
      next: () => {
        this.isLoadingFinance = false;
      },
      error: (error) => {
        console.error('Error loading finance analysis:', error);
        this.isLoadingFinance = false;
      }
    });

    // Load Healthcare Analysis
    this.isLoadingHealthcare = true;
    this.healthcareAnalysis$ = this.systemStatusService.getSectorAnalysis('healthcare');
    this.healthcareAnalysis$.subscribe({
      next: () => {
        this.isLoadingHealthcare = false;
      },
      error: (error) => {
        console.error('Error loading healthcare analysis:', error);
        this.isLoadingHealthcare = false;
      }
    });

    // Load Retail Analysis
    this.isLoadingRetail = true;
    this.retailAnalysis$ = this.systemStatusService.getSectorAnalysis('retail');
    this.retailAnalysis$.subscribe({
      next: () => {
        this.isLoadingRetail = false;
      },
      error: (error) => {
        console.error('Error loading retail analysis:', error);
        this.isLoadingRetail = false;
      }
    });
  }

  refreshSectorAnalysis(sector: string) {
    switch (sector) {
      case 'technology':
        this.isRefreshingTechnology = true;
        this.technologyAnalysis$ = this.systemStatusService.getSectorAnalysis('technology');
        this.technologyAnalysis$.subscribe({
          next: () => {
            this.isRefreshingTechnology = false;
          },
          error: (error) => {
            console.error('Error refreshing technology analysis:', error);
            this.isRefreshingTechnology = false;
          }
        });
        break;
      case 'finance':
        this.isRefreshingFinance = true;
        this.financeAnalysis$ = this.systemStatusService.getSectorAnalysis('finance');
        this.financeAnalysis$.subscribe({
          next: () => {
            this.isRefreshingFinance = false;
          },
          error: (error) => {
            console.error('Error refreshing finance analysis:', error);
            this.isRefreshingFinance = false;
          }
        });
        break;
      case 'healthcare':
        this.isRefreshingHealthcare = true;
        this.healthcareAnalysis$ = this.systemStatusService.getSectorAnalysis('healthcare');
        this.healthcareAnalysis$.subscribe({
          next: () => {
            this.isRefreshingHealthcare = false;
          },
          error: (error) => {
            console.error('Error refreshing healthcare analysis:', error);
            this.isRefreshingHealthcare = false;
          }
        });
        break;
      case 'retail':
        this.isRefreshingRetail = true;
        this.retailAnalysis$ = this.systemStatusService.getSectorAnalysis('retail');
        this.retailAnalysis$.subscribe({
          next: () => {
            this.isRefreshingRetail = false;
          },
          error: (error) => {
            console.error('Error refreshing retail analysis:', error);
            this.isRefreshingRetail = false;
          }
        });
        break;
    }
  }

  refreshAllSectors() {
    this.isRefreshingAll = true;
    this.loadSectorAnalyses();
    
    // Reset all refresh states after a short delay
    setTimeout(() => {
      this.isRefreshingAll = false;
      this.isRefreshingTechnology = false;
      this.isRefreshingFinance = false;
      this.isRefreshingHealthcare = false;
      this.isRefreshingRetail = false;
    }, 5000);
  }

  getCategoryColor(category: string): string {
    switch (category.toLowerCase()) {
      case 'fed':
        return 'bg-blue-100 text-blue-800';
      case 'earnings':
        return 'bg-green-100 text-green-800';
      case 'inflation':
        return 'bg-yellow-100 text-yellow-800';
      case 'geopolitical':
        return 'bg-red-100 text-red-800';
      case 'technology':
        return 'bg-purple-100 text-purple-800';
      case 'energy':
        return 'bg-orange-100 text-orange-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  }

  getTagColor(tag: string): string {
    switch (tag.toLowerCase()) {
      case 'breaking':
        return 'bg-red-100 text-red-800';
      case 'high-impact':
        return 'bg-orange-100 text-orange-800';
      case 'bullish':
        return 'bg-green-100 text-green-800';
      case 'bearish':
        return 'bg-red-100 text-red-800';
      case 'analysis':
        return 'bg-blue-100 text-blue-800';
      case 'technology':
        return 'bg-purple-100 text-purple-800';
      case 'macro':
        return 'bg-indigo-100 text-indigo-800';
      case 'sector':
        return 'bg-cyan-100 text-cyan-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  }

  formatNumber(value: number): string {
    return new Intl.NumberFormat('en-US').format(value);
  }

  formatPercent(value: number): string {
    return `${(value * 100).toFixed(1)}%`;
  }

  getCategoryCounts(documents: any[]): { name: string; count: number }[] {
    const categoryCounts: { [key: string]: number } = {};
    
    documents.forEach(doc => {
      const category = doc.category;
      categoryCounts[category] = (categoryCounts[category] || 0) + 1;
    });

    return Object.entries(categoryCounts).map(([name, count]) => ({
      name,
      count
    }));
  }
}
