#!/bin/bash
# Script to open all HTML visualization files in default browser

echo "Opening HTML visualization files in browser..."
open "xenium_acta2_analysis/acta2_z_level_counts.html"
open "xenium_acta2_analysis/acta2_density_heatmap.html"
open "xenium_acta2_analysis/acta2_z_level_spatial.html"
open "xenium_acta2_analysis/acta2_3d_scatter.html"
open "xenium_dual_gene_analysis/fabp4_expression_heatmap.html"
open "xenium_dual_gene_analysis/acta2_fabp4_3d_scatter.html"
open "xenium_dual_gene_analysis/combined_expression_overlay.html"
open "xenium_dual_gene_analysis/acta2_expression_heatmap.html"
open "xenium_dual_gene_analysis/gene_comparison_counts.html"
open "xenium_dual_gene_analysis/gene_spatial_distribution.html"
open "xenium_expression_filtered_analysis/expression_filtered_3d_visualization.html"
open "xenium_all_transcripts_analysis/all_transcripts_3d_visualization.html"

echo "All files opened. Please capture screenshots manually."
echo "Save screenshots in the 'images' folder with appropriate names."
