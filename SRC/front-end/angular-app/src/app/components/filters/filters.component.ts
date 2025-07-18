import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, FormGroup, FormsModule, ReactiveFormsModule } from '@angular/forms';
import { animate, state, style, transition, trigger } from '@angular/animations';
import { FilterModel, defaultFilters, unitsOptions, targetPredictOptions, modelsOptions } from '../../models/filter.model';
import { PredResultService, PredictionResponse } from '../../services/pred-result.service';
import { PredictionDataFilPredService } from '../../services/prediction-data-fil-pred.service';

@Component({
  selector: 'app-filters',
  standalone: true,
  imports: [CommonModule, FormsModule, ReactiveFormsModule],
  templateUrl: './filters.component.html',
  styleUrls: ['./filters.component.scss'],
  animations: [
    trigger('expandCollapse', [
      state('expanded', style({
        height: '*',
        opacity: 1,
        visibility: 'visible'
      })),
      state('collapsed', style({
        height: '0',
        opacity: 0,
        visibility: 'hidden'
      })),
      transition('expanded <=> collapsed', [
        animate('250ms ease-in-out')
      ])
    ])
  ]
})

export class FiltersComponent implements OnInit {
  filterForm!: FormGroup;
  filtersVisible = true;
  unitsOptions = unitsOptions;
  modelsOptions = modelsOptions;
  targetPredictOptions = targetPredictOptions;

  constructor(private fb: FormBuilder, private PredictionDataFilPredService: PredictionDataFilPredService) {}

  ngOnInit(): void {
    this.initForm();
  }

  initForm(): void {
    this.filterForm = this.fb.group({
      fromDate: [defaultFilters.fromDate],
      toDate: [defaultFilters.toDate],
      unit: [defaultFilters.unit],
      models: [defaultFilters.models],
      targetPredict: [defaultFilters.targetPredict],
      targetValueMax: [defaultFilters.targetValueMax],
      targetValueMin: [defaultFilters.targetValueMin]
    });
  }

  toggleFilters(): void {
    this.filtersVisible = !this.filtersVisible;
  }

  generateReport(): void {
    if (this.filterForm.valid) {
      const filters = this.filterForm.value as FilterModel;
      console.log('Generating report with filters:', filters);

      this.PredictionDataFilPredService.fetchAndStorePredictions(filters);
    }
  }

  resetFilters(): void {
    this.filterForm.reset(defaultFilters);
    this.PredictionDataFilPredService.clearResults();
  }
}