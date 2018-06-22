void copytree(){

    // Read old file and tree
	TFile *oldFile = new TFile("DVntuple_MC16_unweighted.root");
    TTree *oldTree = (TTree*)oldFile->Get("BuTuple/DecayTree");

    // Create new file with compress=101 for ZLIB: 100*algorithm + level
    TFile *newFile = new TFile("DVntuple_MC16_forGBR.root", "recreate", "", 101);
    TTree *newTree = oldTree->CloneTree();

    // Save new file and exit
    newFile->Write();
    delete oldFile;
    delete newFile;


}
