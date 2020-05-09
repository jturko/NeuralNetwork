
void plot_errors_2d(int bins_epoch=10000, double max_epoch=10000., int bins_error=300, double max_error=3.0) {

    TTree * tree = new TTree("tree","tree");
    tree->ReadFile("errors2d.txt","Epoch:Error");
    TH2F * hist = new TH2F("hist","hist",bins_epoch,0.,max_epoch,bins_error,0.,max_error);
    hist->GetXaxis()->SetTitle("Epoch");
    hist->GetYaxis()->SetTitle("Error");
    hist->SetTitle("Error vs. Epoch (2D)");
    float epoch, error;
    tree->SetBranchAddress("Epoch",&epoch);
    tree->SetBranchAddress("Error",&error);
    for(int i=0; i<tree->GetEntries(); i++) {
        tree->GetEntry(i);
        hist->Fill(epoch,error);
    }
    hist->Draw("colz");
}

void plot_errors_1d() {
    TGraph * graph = new TGraph("errors.txt");
    graph->SetTitle("Error vs. Epoch");
    graph->GetXaxis()->SetTitle("Epoch");
    graph->GetYaxis()->SetTitle("Error");
    graph->Draw("a*");
    gPad->SetLogy(true);
}

