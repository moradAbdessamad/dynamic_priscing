import { Component } from '@angular/core';
import { animate, state, style, transition, trigger } from '@angular/animations';
import { NgFor } from '@angular/common';
import { NgIf } from '@angular/common';
import { RouterLink } from '@angular/router';

interface NavItem {
  icon: string;
  label: string;
  active?: boolean;
}
@Component({
  selector: 'app-side-bar',
  standalone: true,
  imports: [NgFor, NgIf, RouterLink],
  templateUrl: './side-bar.component.html',
  styleUrl: './side-bar.component.scss',
  animations: [
    trigger('sidebarAnimation', [
      state('expanded', style({
        width: '260px'
      })),
      state('collapsed', style({
        width: '72px'
      })),
      transition('expanded <=> collapsed', [
        animate('300ms ease')
      ])
    ])
  ]
})
export class SideBarComponent {
  sidebarExpanded = true;

  navItems: NavItem[] = [
    { icon: 'Predict Range Date', label: '' },
    { icon: 'Predict Single Date', label: 'single-date-pred'},
  ];

  toggleSidebar() {
    this.sidebarExpanded = !this.sidebarExpanded;
  }
}
