import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SystemStatusService, RiskAnalysis, RiskMetrics, MarketRisk, RiskAlert } from '../../services/system-status.service';
import { Observable } from 'rxjs';

@Component({
  selector: 'app-risk',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './risk.component.html',
  styles: []
})
export class RiskComponent implements OnInit {
  riskAnalysis$: Observable<RiskAnalysis> | undefined;
  riskMetrics$: Observable<RiskMetrics> | undefined;
  marketRisk$: Observable<MarketRisk> | undefined;
  riskAlerts$: Observable<RiskAlert[]> | undefined;

  constructor(private systemStatusService: SystemStatusService) {}

  ngOnInit() {
    this.riskAnalysis$ = this.systemStatusService.getRiskAnalysis();
    this.riskMetrics$ = this.systemStatusService.getRiskMetrics();
    this.marketRisk$ = this.systemStatusService.getMarketRisk();
    this.riskAlerts$ = this.systemStatusService.getRiskAlerts();
  }
}
