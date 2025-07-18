import { Injectable } from "@angular/core";
import { BehaviorSubject, Observable, throwError } from "rxjs";
import { tap, catchError, map } from "rxjs/operators";
import { FilterModel } from "../models/filter.model";
import {
  PredResultService,
  PredictionResponse,
  ParsedPredictionResponse,
  PredictionResult,
} from "./pred-result.service";

@Injectable({
  providedIn: "root",
})
export class PredictionDataFilPredService {
  private predictionResultSubject =
    new BehaviorSubject<ParsedPredictionResponse | null>(null);
  public predictionResult$: Observable<ParsedPredictionResponse | null> =
    this.predictionResultSubject.asObservable();

  private isLoadingSubject = new BehaviorSubject<boolean>(false);
  public isLoading$: Observable<boolean> = this.isLoadingSubject.asObservable();

  private errorSubject = new BehaviorSubject<any | null>(null);
  public error$: Observable<any | null> = this.errorSubject.asObservable();

  constructor(private predResultService: PredResultService) {}

  fetchAndStorePredictions(filters: FilterModel): void {
    this.isLoadingSubject.next(true);
    this.errorSubject.next(null);
    this.predictionResultSubject.next(null);

    this.predResultService
      .getPrediction(filters)
      .pipe(
        // No need to parse response.result as it's already an object
        tap((response: PredictionResponse) => {
          this.predictionResultSubject.next(response);
          this.isLoadingSubject.next(false);
        }),
        catchError((error) => {
          console.error("Error fetching predictions in shared service:", error);
          this.errorSubject.next(error);
          this.predictionResultSubject.next(null);
          this.isLoadingSubject.next(false);
          return throwError(() => error);
        })
      )
      .subscribe();
  }

  clearResults(): void {
    this.predictionResultSubject.next(null);
    this.errorSubject.next(null);
    this.isLoadingSubject.next(false);
  }
}
