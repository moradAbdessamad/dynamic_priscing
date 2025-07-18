import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subscription } from 'rxjs';
import { SinglePredictionDataFilPredService } from '../../services/single-prediction-data-fil-pred.service';
import { SingleDatePredictionResponse } from '../../services/single-pred-result.service';
import { DynamicTableComponent, TableData } from '../dynamic-table/dynamic-table.component';

@Component({
  selector: 'app-single-pred-result',
  standalone: true,
  imports: [CommonModule, DynamicTableComponent],
  templateUrl: './single-pred-result.component.html',
  styleUrl: './single-pred-result.component.scss'
})
export class SinglePredResultComponent implements OnInit, OnDestroy {
  predictionData: SingleDatePredictionResponse | null = null;
  isLoading = false;
  error: any = null;
  hasData = false;
  
  private subscription = new Subscription();
  
  get tableData(): TableData {
    if (!this.predictionData?.result) {
      return {};
    }
    
    return this.predictionData.result;
  }
  
  constructor(private singlePredictionDataFilPredService: SinglePredictionDataFilPredService) {}
  
  ngOnInit(): void {
    this.subscription.add(
      this.singlePredictionDataFilPredService.isLoading$.subscribe(
        isLoading => this.isLoading = isLoading
      )
    );
    
    this.subscription.add(
      this.singlePredictionDataFilPredService.error$.subscribe(
        error => this.error = error
      )
    );
    
    this.subscription.add(
      this.singlePredictionDataFilPredService.predictionResult$.subscribe(
        result => {
          this.predictionData = result;
          this.hasData = !!result?.result;
          console.log('Single prediction data received:', result);
        }
      )
    );
  }
  
  ngOnDestroy(): void {
    this.subscription.unsubscribe();
  }
}