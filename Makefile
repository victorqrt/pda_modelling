all: env
	chmod +x run

env:
	virtualenv -p python3 env
	. env/bin/activate && pip install scikit-learn pandas matplotlib

clean:
	chmod -x run
	rm -fr env __pycache__ model.bin plots
