import { Component } from '@angular/core';
import { FiltersComponent } from '../filters/filters.component';
import { SingleFiltersComponent } from "../single-filters/single-filters.component";
import { SinglePredResultComponent } from '../single-pred-result/single-pred-result.component';
import { PredResultComponent } from "../pred-result/pred-result.component";

@Component({
  selector: 'app-single-dashboard',
  imports: [SingleFiltersComponent, SinglePredResultComponent],
  templateUrl: './single-dashboard.component.html',
  styleUrl: './single-dashboard.component.scss'
})
export class SingleDashboardComponent {

}
