void copytree(){

    // Read old file and tree
    TFile *oldFile = new TFile("folding_weights.root");
    TTree *oldTree = (TTree*)oldFile->Get("reweighting");

    // Create new file with compress=101 for ZLIB: 100*algorithm + level
    TFile *newFile = new TFile("new.root", "recreate", "", 101);
    TTree *newTree = oldTree->CloneTree();

    // Save new file and exit
    newFile->Write();
    delete oldFile;
    delete newFile;


}
