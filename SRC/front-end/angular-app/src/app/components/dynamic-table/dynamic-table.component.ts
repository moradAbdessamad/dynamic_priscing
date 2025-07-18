import { Component, Input, OnChanges } from '@angular/core';
import { CommonModule } from '@angular/common';

export interface TableData {
  [date: string]: {
    [metric: string]: number; 
  };
}

@Component({
  selector: 'app-dynamic-table',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './dynamic-table.component.html',
  styleUrls: ['./dynamic-table.component.scss']
})

export class DynamicTableComponent implements OnChanges {
  @Input() data: TableData = {};

  dates: string[] = [];
  metricKeys: string[] = [];
  rows: { label: string, values: number[] }[] = [];

  ngOnChanges() {
    this.processData();
  }

  private processData() {
    this.dates = Object.keys(this.data).sort();

    const firstDateEntry = this.data[this.dates[0]];
    this.metricKeys = firstDateEntry ? Object.keys(firstDateEntry) : [];

    this.rows = this.metricKeys.map(metric => ({
      label: this.formatLabel(metric),
      values: this.dates.map(date => this.data[date]?.[metric] ?? 0)
    }));
  }

  formatLabel(label: string): string {
    return label
      .replace(/_/g, ' ')
      .replace(/\b\w/g, char => char.toUpperCase());
  }

  formatDate(dateStr: string): string {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    });
  }

  formatNumber(value: number): string {
    return value.toLocaleString(undefined, {
      minimumFractionDigits: 0,
      maximumFractionDigits: 2
    });
  }
}
