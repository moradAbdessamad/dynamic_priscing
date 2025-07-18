import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { SingleDateFilterModel } from '../models/singleFilter.model';

export interface ApiSingleDatePredictionRequest {
  singleDate: string;
  models: string;
  targetPredict: string;
  targetValueMin: number;
  targetValueMax: number;
  unit: string;
}

export interface PredictionResult {
  [date: string]: {
    best_input_value: number;
    max_predicted_value: number;
  };
}

export interface SingleDatePredictionResponse {
  model: string;
  result: PredictionResult;
  status: string;
  target_type: string;
  unit: string;
}

@Injectable({
  providedIn: 'root'
})

export class SinglePredResultService {

  private apiUrl = 'http://localhost:5300/api/calculate_predicted_single_date';

  constructor(private http: HttpClient) { }

  getSingleDatePrediction(filterData: SingleDateFilterModel): Observable<SingleDatePredictionResponse> {
    const apiRequestData: ApiSingleDatePredictionRequest = {
      ...filterData,
      unit: filterData.unit.toString()
    };
    return this.http.post<SingleDatePredictionResponse>(this.apiUrl, apiRequestData);
  }
}
