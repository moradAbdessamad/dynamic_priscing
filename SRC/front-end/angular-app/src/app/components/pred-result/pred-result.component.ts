import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule, JsonPipe } from '@angular/common';
import { Subscription } from 'rxjs';
import { PredictionDataFilPredService } from '../../services/prediction-data-fil-pred.service';
import { ParsedPredictionResponse, PredictionResult } from '../../services/pred-result.service';
import { DynamicTableComponent } from '../dynamic-table/dynamic-table.component';
import { TableData } from '../dynamic-table/dynamic-table.component'; 

@Component({
  selector: 'app-pred-result',
  standalone: true,
  imports: [CommonModule, DynamicTableComponent],
  templateUrl: './pred-result.component.html',
  styleUrls: ['./pred-result.component.scss']
})

export class PredResultComponent implements OnInit, OnDestroy {
  predictionData: ParsedPredictionResponse | null = null;
  isLoading: boolean = true;
  error: any | null = null;
  hasData: boolean = false;

  private subscriptions = new Subscription();

  get tableData(): TableData {
    return this.predictionData?.result || {};
  }

  constructor(private predictionDataFilPredService: PredictionDataFilPredService) {}

  ngOnInit(): void {
    this.subscriptions.add(
      this.predictionDataFilPredService.predictionResult$.subscribe(data => {
        this.predictionData = data;
        this.hasData = !!data?.result;
        console.log('Prediction data received:', data);
      })
    );
    
    this.subscriptions.add(
      this.predictionDataFilPredService.isLoading$.subscribe(loading => {
        console.log('Prediction request in progress...');
        this.isLoading = loading;
      })
    );
    
    this.subscriptions.add(
      this.predictionDataFilPredService.error$.subscribe(err => {
        this.error = err;
        console.error('Prediction request error:', err);
      })
    );
  }

  ngOnDestroy(): void {
    this.subscriptions.unsubscribe();
  }
}