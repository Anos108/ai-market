import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { ModalService } from '../../shared/modal/modal.service';

@Component({
  selector: 'app-symbol-management',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './symbol-management.component.html',
  styleUrls: ['./symbol-management.component.css']
})
export class SymbolManagementComponent implements OnInit {
  loading = true;
  error: string | null = null;
  summary: any = null;
  symbols: any[] = [];
  searchResults: any[] = [];
  tradingDecisions: any[] = [];
  
  adding = false;
  loadingDecisions = false;
  refreshing = false;
  searchQuery = '';
  statusFilter = '';
  
  newSymbol = {
    symbol: '',
    name: '',
    sector: '',
    industry: '',
    status: 'active',
    priority: 3,
    notes: ''
  };

  constructor(private http: HttpClient, private modalService: ModalService) {}

  ngOnInit(): void {
    this.loadSummary();
    this.loadSymbols();
    this.loadTradingDecisions();
  }

  loadSummary(): void {
    this.loading = true;
    this.error = null;

    this.http.get<any>('http://localhost:8001/symbols/summary').subscribe({
      next: (summary) => {
        this.summary = summary;
        this.loading = false;
      },
      error: (err) => {
        console.error('Error loading summary:', err);
        this.error = 'Failed to load symbol management summary.';
        this.loading = false;
      }
    });
  }

  loadSymbols(): void {
    this.refreshing = true;
    
    let url = 'http://localhost:8001/api/symbols';
    if (this.statusFilter) {
      url += `?status=${this.statusFilter}`;
    }

    console.log('Refreshing symbols from:', url);
    this.http.get<any>(url).subscribe({
      next: (response) => {
        console.log('Symbols loaded:', response.symbols?.length || 0);
        this.symbols = response.symbols || [];
        this.error = null; // Clear any previous errors
        this.refreshing = false;
      },
      error: (err) => {
        console.error('Error loading symbols:', err);
        this.error = 'Failed to load symbols.';
        this.symbols = []; // Clear symbols on error
        this.refreshing = false;
      }
    });
  }

  addSymbol(): void {
    if (!this.newSymbol.symbol || !this.newSymbol.name || !this.newSymbol.sector || !this.newSymbol.industry) {
      this.modalService.warning('Please fill in all required fields.');
      return;
    }

    this.adding = true;

    this.http.post<any>(`http://localhost:8001/api/symbols`, {
      symbol: this.newSymbol.symbol,
      name: this.newSymbol.name,
      sector: this.newSymbol.sector,
      industry: this.newSymbol.industry,
      status: this.newSymbol.status,
      priority: this.newSymbol.priority,
      notes: this.newSymbol.notes
    }).subscribe({
      next: (response) => {
        this.adding = false;
        this.modalService.success(`Symbol ${this.newSymbol.symbol} added successfully!`);
        this.resetForm();
        this.loadSummary();
        this.loadSymbols();
      },
      error: (err) => {
        console.error('Error adding symbol:', err);
        this.adding = false;
        this.modalService.error('Failed to add symbol. Please try again.');
      }
    });
  }

  addFromSearch(result: any): void {
    this.newSymbol.symbol = result.symbol;
    this.newSymbol.name = result.name;
    this.newSymbol.sector = result.sector;
    this.newSymbol.industry = result.industry;
    this.addSymbol();
  }

  removeSymbol(symbol: string): void {
    this.modalService.confirm(
      `Are you sure you want to remove ${symbol} from your portfolio?`,
      'Remove Symbol',
      'Remove',
      'Cancel'
    ).then((confirmed) => {
      if (confirmed) {
        this.http.delete<any>(`http://localhost:8001/api/symbols/${symbol}`).subscribe({
          next: (response) => {
            this.modalService.success(`Symbol ${symbol} removed successfully!`);
            this.loadSummary();
            this.loadSymbols();
          },
          error: (err) => {
            console.error('Error removing symbol:', err);
            this.modalService.error('Failed to remove symbol. Please try again.');
          }
        });
      }
    });
  }

  searchSymbols(): void {
    if (!this.searchQuery.trim()) {
      this.searchResults = [];
      return;
    }

    this.http.get<any[]>(`http://localhost:8001/symbols/search?query=${encodeURIComponent(this.searchQuery)}`).subscribe({
      next: (results) => {
        this.searchResults = results;
      },
      error: (err) => {
        console.error('Error searching symbols:', err);
        this.searchResults = [];
      }
    });
  }

  loadTradingDecisions(): void {
    this.loadingDecisions = true;

    this.http.get<any>('http://localhost:8001/symbols/trading-decisions').subscribe({
      next: (decisions) => {
        // The API returns an array directly, not an object with property values
        this.tradingDecisions = Array.isArray(decisions) ? decisions : [];
        this.loadingDecisions = false;
      },
      error: (err) => {
        console.error('Error loading trading decisions:', err);
        this.loadingDecisions = false;
        this.tradingDecisions = [];
      }
    });
  }

  viewSymbolDetails(symbol: any): void {
    this.modalService.info(`View details for ${symbol.symbol}`, 'Symbol Details');
  }

  editSymbol(symbol: any): void {
    this.modalService.info(`Edit functionality coming soon for ${symbol.symbol}`, 'Edit Symbol');
  }

  resetForm(): void {
    this.newSymbol = {
      symbol: '',
      name: '',
      sector: '',
      industry: '',
      status: 'active',
      priority: 3,
      notes: ''
    };
  }

  getStatusClass(status: string): string {
    switch (status) {
      case 'active':
        return 'bg-green-100 text-green-800';
      case 'monitoring':
        return 'bg-yellow-100 text-yellow-800';
      case 'watchlist':
        return 'bg-purple-100 text-purple-800';
      case 'inactive':
        return 'bg-gray-100 text-gray-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  }

  getActionClass(action: string): string {
    const actionLower = action.toLowerCase();
    switch (actionLower) {
      case 'buy':
        return 'bg-green-100 text-green-800';
      case 'sell':
        return 'bg-red-100 text-red-800';
      case 'hold':
        return 'bg-blue-100 text-blue-800';
      case 'watch':
        return 'bg-yellow-100 text-yellow-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  }
}
