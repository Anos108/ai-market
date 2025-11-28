import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

export interface ModalConfig {
  title: string;
  message: string;
  type: 'info' | 'success' | 'warning' | 'error' | 'confirm';
  confirmText?: string;
  cancelText?: string;
  onConfirm?: () => void;
  onCancel?: () => void;
  showCancel?: boolean;
}

@Injectable({
  providedIn: 'root'
})
export class ModalService {
  private modalSubject = new BehaviorSubject<ModalConfig | null>(null);
  public modal$ = this.modalSubject.asObservable();

  show(config: ModalConfig) {
    this.modalSubject.next(config);
  }

  hide() {
    this.modalSubject.next(null);
  }

  // Convenience methods
  success(message: string, title: string = 'Success') {
    this.show({ title, message, type: 'success', confirmText: 'OK' });
  }

  error(message: string, title: string = 'Error') {
    this.show({ title, message, type: 'error', confirmText: 'OK' });
  }

  warning(message: string, title: string = 'Warning') {
    this.show({ title, message, type: 'warning', confirmText: 'OK' });
  }

  info(message: string, title: string = 'Information') {
    this.show({ title, message, type: 'info', confirmText: 'OK' });
  }

  confirm(
    message: string, 
    title: string = 'Confirm', 
    confirmText: string = 'Confirm',
    cancelText: string = 'Cancel'
  ): Promise<boolean> {
    return new Promise<boolean>((resolve) => {
      this.show({
        title,
        message,
        type: 'confirm',
        confirmText,
        cancelText,
        showCancel: true,
        onConfirm: () => {
          this.hide();
          resolve(true);
        },
        onCancel: () => {
          this.hide();
          resolve(false);
        }
      });
    });
  }
}
