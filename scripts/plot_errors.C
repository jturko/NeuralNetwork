
void plot_errors() {
    TGraph * graph = new TGraph("errors.txt");
    graph->Draw("a*");
}

