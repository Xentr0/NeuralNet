all: clean NeuralNet

%.o: %.cpp
	g++ -o $@ -c $<

NeuralNet: Main.cpp Neuron.o Net.o Handler.o
	g++ -o $@ $^

clean:
	rm -f *.o
