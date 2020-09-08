AUX=00_aux
MTL_AM=01_mtl-argmin
BILING=02_bilingual-epistemic-segmentation
MULTI_DOMAIN=03_multi-domain-argmin-boundary-detection
SEQ_2_SEQ=04_seq-2-seq

all:
	cd $(AUX) && make
	cd $(MTL_AM) && make
	cd $(BILING) && make
	cd $(MULTI_DOMAIN) && make
	cd $(SEQ_2_SEQ) && make

clean:
	cd $(AUX) && make clean
	cd $(MTL_AM) && make clean
	cd $(BILING) && make clean
	cd $(MULTI_DOMAIN) && make clean
	cd $(SEQ_2_SEQ) && make clean
