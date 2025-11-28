import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Observable } from 'rxjs';
import { SystemStatusService, RLStrategyAgentSummary } from '../../services/system-status.service';

@Component({
  selector: 'app-rl-strategy-agent',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './rl-strategy-agent.component.html',
  styles: []
})
export class RLStrategyAgentComponent implements OnInit {
  rlStrategyAgentSummary$: Observable<RLStrategyAgentSummary> | undefined;
  rlTrainingStatus$: Observable<any> | undefined;
  rlPerformance$: Observable<any> | undefined;
  rlActions$: Observable<any> | undefined;

  // Make Object available in template
  Object = Object;

  constructor(private systemStatusService: SystemStatusService) {}

  ngOnInit() {
    this.rlStrategyAgentSummary$ = this.systemStatusService.getRLStrategyAgentSummary();
    this.rlTrainingStatus$ = this.systemStatusService.getRLTrainingStatus();
    this.rlPerformance$ = this.systemStatusService.getRLPerformance();
    this.rlActions$ = this.systemStatusService.getRLActions();
  }
}
