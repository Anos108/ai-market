import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SystemStatusService, ExecutionAgentSummary } from '../../services/system-status.service';
import { Observable } from 'rxjs';

@Component({
  selector: 'app-execution-agent',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './execution-agent.component.html',
  styles: []
})
export class ExecutionAgentComponent implements OnInit {
  executionAgentSummary$: Observable<ExecutionAgentSummary> | undefined;

  constructor(private systemStatusService: SystemStatusService) {}

  ngOnInit() {
    this.executionAgentSummary$ = this.systemStatusService.getExecutionAgentSummary();
  }
}