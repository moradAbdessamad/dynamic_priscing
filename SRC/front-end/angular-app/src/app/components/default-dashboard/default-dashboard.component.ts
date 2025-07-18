import { Component } from '@angular/core';
import { FiltersComponent } from '../filters/filters.component';
import { PredResultComponent } from '../pred-result/pred-result.component';
import { DynamicTableComponent } from '../dynamic-table/dynamic-table.component';

@Component({
  selector: 'app-default-dashboard',
  imports: [FiltersComponent, PredResultComponent],
  templateUrl: './default-dashboard.component.html',
  styleUrl: './default-dashboard.component.scss'
})
export class DefaultDashboardComponent {

}
