package com.example.lettergrabbermecacucsur;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;


import android.Manifest;
import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
//import android.content.res.AssetManager;
import android.os.Bundle;

import android.util.Log;
import android.util.Pair;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
//import java.io.InputStream;
//import java.nio.ByteBuffer;
//import java.nio.ByteOrder;
//import java.nio.MappedByteBuffer;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.List;

///import org.apache.commons.math3.analysis.function.Mean;
//import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;

//import java.util.PriorityQueue;
//import org.tensorflow.lite.Interpreter;
//import org.tensorflow.lite.DataType;
//import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

public class MainActivity extends AppCompatActivity {

    private boolean isDrawing = false;
    private List<Pair<Float, Float>> coordinates = new ArrayList<>();
    private List<Long> timeOffsets = new ArrayList<>();
    private List<Float> pressures = new ArrayList<>();
    private static final int REQUEST_WRITE_EXTERNAL_STORAGE = 1;

    //private Interpreter tflite;
    //private float[][] outputValues;
    // Create a container for the result and specify that this is a quantized model.
    // Hence, the 'DataType' is defined as UINT8 (8-bit unsigned integer)
    //TensorBuffer probabilityBuffer =
    //        TensorBuffer.createFixedSize(new int[]{1, 12}, DataType.UINT8);

    //private boolean isDarkTheme = false; // Por defecto, asumimos que estamos en modo claro
    private DrawingView drawingView; // Declaración a nivel de clase

    @SuppressLint("ClickableViewAccessibility")
    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        drawingView = findViewById(R.id.drawingCanvas);

        Button captureButton = findViewById(R.id.capture);
        Button clearButton = findViewById(R.id.clear);
        //Button predictButton = findViewById(R.id.predict);

        // Cargar el modelo TensorFlow Lite
        /*
        try {
            // Load the model from the assets directory
            tflite = new Interpreter(loadModelFile("model.tflite"));

        } catch (IOException e) {
            e.printStackTrace();
            Log.d("TAG", "tflite error: " + e.getMessage());


        }
        */



        captureButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (checkPermission()) {
                    captureData();
                } else {
                    requestPermission();
                }
            }
        });

        clearButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                drawingView.clearDrawing();
                coordinates.clear();
                timeOffsets.clear(); // Limpiar la lista de tiempos
                pressures.clear();

            }
        });
/*
        predictButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (checkPermission()) {
                    /*
                    // Get the width and height of the DrawingView
                    int viewWidth = drawingView.getWidth();
                    int viewHeight = drawingView.getHeight();

                    // Create a Bitmap of the same size as the DrawingView
                    Bitmap capturedBitmap = Bitmap.createBitmap(viewWidth, viewHeight, Bitmap.Config.ARGB_8888);

                    // Create a Canvas to draw onto the Bitmap
                    Canvas canvas = new Canvas(capturedBitmap);

                    // Draw the contents of the DrawingView onto the Bitmap
                    drawingView.draw(canvas);

                    // Resize the captured image to 227x227 pixels
                    int targetWidth = 227;
                    int targetHeight = 400;
                    Bitmap resizedBitmap = Bitmap.createScaledBitmap(capturedBitmap, targetWidth, targetHeight, true);
                    ////
                    // Now you have a resized image (resizedBitmap) with dimensions 227x227 pixels

                    try {
                        Log.d("TAG", "tflite error try: ");
                        if (timeOffsets.size()!=0 && coordinates.size()!=0 && pressures.size()!=0) {
                            Log.d("TAG", "tflite error no entrar: ");

                            // Obtén el primer tiempo en milisegundos

                            long firstTime = timeOffsets.get(0);

                            // Recorre y ajusta los tiempos
                            List<Float> adjustedTimeOffsets = new ArrayList<>();
                            for (long timeOffset : timeOffsets) {
                                long adjustedTime = timeOffset - firstTime;
                                adjustedTimeOffsets.add(Float.valueOf(adjustedTime));
                            }
                            List<Float> coordinates_x = new ArrayList<>();
                            List<Float> coordinates_y = new ArrayList<>();
                            List<Float> pressuresArray = new ArrayList<>();

                            for (int index = 0; index < coordinates.size(); index++) {

                                Pair<Float, Float> coordinate = coordinates.get(index);
                                coordinates_x.add(coordinate.first);
                                coordinates_y.add(coordinate.second);
                                pressuresArray.add(pressures.get(index));

                            }


                            predict(extractFeatures(convertListToArray(coordinates_x),convertListToArray(coordinates_y),convertListToArray(adjustedTimeOffsets),convertListToArray(pressuresArray)));
                        }else{
                            Toast.makeText(null, "Ningún dato obtenido (Dibuja una letra).", Toast.LENGTH_SHORT).show();

                        }
                    }catch (RuntimeException e){
                        e.printStackTrace();
                        Toast.makeText(null, "Error al predecir: " + e.getMessage(), Toast.LENGTH_SHORT).show();

                    }



                } else {
                    requestPermission();
                }
            }
        });
*/

        drawingView.setOnTouchListener(new View.OnTouchListener() {

            @Override
            public boolean onTouch(View v, MotionEvent event) {
                switch (event.getAction()) {
                    case MotionEvent.ACTION_DOWN:
                        isDrawing = true;
                        coordinates.add(new Pair<>(event.getX(), drawingView.getHeight() - event.getY())); // Invertir las coordenadas en el eje Y
                        timeOffsets.add(System.currentTimeMillis());
                        pressures.add(event.getPressure());
                        drawingView.addPoint(event.getX(), event.getY());
                        break;
                    case MotionEvent.ACTION_MOVE:
                        if (isDrawing) {
                            coordinates.add(new Pair<>(event.getX(),  drawingView.getHeight() - event.getY()));
                            timeOffsets.add(System.currentTimeMillis());
                            pressures.add(event.getPressure());
                            drawingView.addPoint(event.getX(), event.getY());
                        }
                        break;
                    case MotionEvent.ACTION_UP:
                        //isDrawing = false;
                        timeOffsets.add(System.currentTimeMillis());
                        coordinates.add(new Pair<>(Float.NaN, Float.NaN));
                        pressures.add(Float.NaN);
                        drawingView.invalidate();
                        break;
                }
                //Log.d("TAG", "Presion tamaño: " + pressures.size() + " valor de presion: " + event.getPressure());

                return true;

            }

        });


    }
/*
    public static float[] convertListToArray(List<Float> floatList) {
        float[] floatArray = new float[floatList.size()];
        for (int i = 0; i < floatList.size(); i++) {
            floatArray[i] = floatList.get(i);
        }
        return floatArray;
    }
*/

    // Check if the WRITE_EXTERNAL_STORAGE permission is granted
    private boolean checkPermission() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED;
    }

    // Request the WRITE_EXTERNAL_STORAGE permission
    private void requestPermission() {
        ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, REQUEST_WRITE_EXTERNAL_STORAGE);
    }

    private void saveDataToFile(String name, String grade) {
        if (checkPermission()) {

            try {

                if (timeOffsets.size()!=0){

                    DateFormat df = new SimpleDateFormat("dd_MM_yyyy_HH_mm_ss");
                    String date = df.format(Calendar.getInstance().getTime());

                    String fileName = name.toUpperCase() + "+" + grade + "-" + date + ".txt";//+ "-" + studentCode.toUpperCase()
                    File folder = new File(getExternalFilesDir(null), "Data/"); //+name.toUpperCase()+"-"+grade
                    folder.mkdirs();

                    File file = new File(folder, fileName);
                    file.createNewFile();

                    //long currentTime = System.currentTimeMillis();
                    StringBuilder content = new StringBuilder();
                    content.append("Time,X,Y,P\n");
                    //Log.d("TAG", "Tamaño coordenadas: " + coordinates.size() + " Tamaño tiempo: " + timeOffsets.size());


                    // Obtén el primer tiempo en milisegundos
                    long firstTime = timeOffsets.get(0);

                    // Recorre y ajusta los tiempos
                    List<Long> adjustedTimeOffsets = new ArrayList<>();
                    for (long timeOffset : timeOffsets) {
                        long adjustedTime = timeOffset - firstTime;
                        adjustedTimeOffsets.add(adjustedTime);
                    }

                    for (int index = 0; index < coordinates.size(); index++) {
                        Pair<Float, Float> coordinate = coordinates.get(index);
                        long timeOffset = adjustedTimeOffsets.get(index); // Obtener el tiempo relativo correspondiente
                        Float pressure = pressures.get(index);
                        content.append(timeOffset).append(",").append(coordinate.first).append(",").append(coordinate.second).append(",").append(pressure).append("\n"); //
                    }
                    FileWriter writer = new FileWriter(file);
                    writer.write(content.toString());
                    writer.close();

                    Toast.makeText(this, "Datos capturados y guardados. \nCantidad: "+adjustedTimeOffsets.size() + " puntos.", Toast.LENGTH_SHORT).show();

                    //Toast toast = Toast.makeText(this, "Datos capturados y guardados. \nCantidad: "+adjustedTimeOffsets.size() + " puntos.", Toast.LENGTH_SHORT);
                    //TextView v = (TextView) toast.getView().findViewById(android.R.id.message);
                    //if( v != null) v.setGravity(Gravity.CENTER);
                    //toast.show();

                    //Toast.makeText(this, , Toast.LENGTH_SHORT).show();
                    adjustedTimeOffsets.clear();
                    drawingView.clearDrawing();
                    coordinates.clear();
                    timeOffsets.clear();
                    pressures.clear();
                }else{
                    Toast.makeText(this, "Ningún dato guardado (Dibuja una letra).", Toast.LENGTH_SHORT).show();
                }



            } catch (IOException e) {
                e.printStackTrace();
                Toast.makeText(this, "Error al guardar archivo: " + e.getMessage(), Toast.LENGTH_SHORT).show();
            }catch (RuntimeException e){
                e.printStackTrace();
                Toast.makeText(this, "Error despues de guardar el archivo: " + e.getMessage(), Toast.LENGTH_SHORT).show();
            }
        } else {
            requestPermission();
        }
    }

    private void captureData() {
        EditText nameEditText = findViewById(R.id.nameEditText);
        EditText gradeEditText = findViewById(R.id.gradeEditText);
        //EditText studentCodeEditText = findViewById(R.id.studentCodeEditText);

        String name = nameEditText.getText().toString().trim();
        String grade = gradeEditText.getText().toString().trim();
        //String studentCode = studentCodeEditText.getText().toString().trim();

        if (name.isEmpty() || grade.isEmpty()) {//studentCode.isEmpty()
            Toast.makeText(this, "Por favor, complete todos los campos.", Toast.LENGTH_SHORT).show();
            return;
        }

        saveDataToFile(name, grade);

    }

    // Handle permission request results
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_WRITE_EXTERNAL_STORAGE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                captureData();
            } else {
                Toast.makeText(this, "Permission denied.", Toast.LENGTH_SHORT).show();
            }
        }
    }

    //+++
/*
    // Función para calcular la derivada con respecto al tiempo
    private double[] calculateDerivative(float[] data, float[] time) {
        int n = data.length;
        double[] derivative = new double[n];

        for (int i = 1; i < n; i++) {
            derivative[i] = (data[i] - data[i - 1]) / (time[i] - time[i - 1]);
        }

        // Para el valor inicial, simplemente copia el valor del segundo punto
        derivative[0] = derivative[1];

        return derivative;
    }

    //Mean
    public static double calculateMean(double[] numbers) {
        if (numbers == null || numbers.length == 0) {
            throw new IllegalArgumentException("Input array is empty or null.");
        }

        double sum = 0.0;
        for (double number : numbers) {
            if (!Double.isNaN(number) && Double.isFinite(number)) {
                sum += number;
            }

        }

        return sum / numbers.length;
    }

    // Función para calcular la media y desviación estándar de un arreglo de datos

    private static float[] calculateMeanAndMad(double[] data) {
        //Mean meanCalculator = new Mean();
        double meanValue = calculateMean(data);

        // Calculate Mean Absolute Deviation (MAD)
        double madSum = 0;
        for (double value : data) {
            if (!Double.isNaN(value) && Double.isFinite(value)) {
                madSum += Math.abs(value - meanValue);
            }
        }
        double madValue = madSum / data.length;

        // Convert mean and MAD to float
        float mean = (float) meanValue;
        float mad = (float) madValue;
        Log.d("TAG", "mean and mad: " + mean +" "+mad);
        return new float[]{mean, mad};
    }


    // Función para calcular todas las correlaciones posibles
    private double[] calculateCorrelations(double[] x, double[] y, double[] pressure, double[] dXdT, double[] dYdT) {
        int numVariables = 5; // X, Y, Pressure, dXdT, dYdT
        int numCorrelations = numVariables * (numVariables - 1) / 2; // Número de combinaciones únicas

        double[] correlations = new double[numCorrelations];
        int index = 0;

        PearsonsCorrelation pearsonsCorrelation = new PearsonsCorrelation();

        // Calcula las correlaciones para todas las combinaciones posibles
        correlations[index++] = pearsonsCorrelation.correlation(x, y);
        correlations[index++] = pearsonsCorrelation.correlation(x, pressure);
        correlations[index++] = pearsonsCorrelation.correlation(x, dXdT);
        correlations[index++] = pearsonsCorrelation.correlation(x, dYdT);
        correlations[index++] = pearsonsCorrelation.correlation(y, pressure);
        correlations[index++] = pearsonsCorrelation.correlation(y, dXdT);
        correlations[index++] = pearsonsCorrelation.correlation(y, dYdT);
        correlations[index++] = pearsonsCorrelation.correlation(pressure, dXdT);
        correlations[index++] = pearsonsCorrelation.correlation(pressure, dYdT);
        correlations[index++] = pearsonsCorrelation.correlation(dXdT, dYdT);

        return correlations;
    }


    // Función para calcular los mínimos locales
    private int countLocalMinima(double[] data, float minProminence) {
        int n = data.length;
        int localMinCount = 0;
        boolean isDescending = true;
        List<Double> localMinima = new ArrayList<>();

        for (int i = 1; i < n - 1; i++) {
            if (data[i] < data[i - 1] && data[i] < data[i + 1]) {
                localMinCount++;
                localMinima.add(data[i]);
                isDescending = true;
            } else if (isDescending && data[i] < data[i - 1]) {
                // Descartar pequeños descensos antes de un mínimo local
                while (i < n - 1 && data[i] < data[i - 1]) {
                    i++;
                }
            } else {
                isDescending = false;
            }
        }

        // Filtrar mínimos locales con prominencia menor que minProminence * derivScale
        float threshold = minProminence;
        int validLocalMinCount = 0;
        for (double localMin : localMinima) {
            if (localMin < threshold) {
                validLocalMinCount++;
            }
        }

        return validLocalMinCount;
    }

    // Función para calcular los máximos locales
    private int countLocalMaxima(double[] data, float minProminence) {
        int n = data.length;
        int localMaxCount = 0;
        boolean isAscending = true;
        List<Double> localMaxima = new ArrayList<>();

        for (int i = 1; i < n - 1; i++) {
            if (data[i] > data[i - 1] && data[i] > data[i + 1]) {
                localMaxCount++;
                localMaxima.add(data[i]);
                isAscending = true;
            } else if (isAscending && data[i] > data[i - 1]) {
                // Descartar pequeños ascensos antes de un máximo local
                while (i < n - 1 && data[i] > data[i - 1]) {
                    i++;
                }
            } else {
                isAscending = false;
            }
        }

        // Filtrar máximos locales con prominencia menor que minProminence * derivScale
        float threshold = minProminence;
        int validLocalMaxCount = 0;
        for (double localMax : localMaxima) {
            if (localMax > threshold) {
                validLocalMaxCount++;
            }
        }

        return validLocalMaxCount;
    }

    // Define a utility function to convert float[] to double[]
    private static double[] floatArrayToDoubleArray(float[] floatArray) {
        double[] doubleArray = new double[floatArray.length];
        for (int i = 0; i < floatArray.length; i++) {
            doubleArray[i] = floatArray[i];
        }
        return doubleArray;
    }
    // Define a utility function to convert double[] to float[]
    private static float[] doubleArrayToFloatArray(double[] doubleArray) {
        float[] floatArray = new float[doubleArray.length];
        for (int i = 0; i < doubleArray.length; i++) {
            floatArray[i] = (float) doubleArray[i];
        }
        return floatArray;
    }


    // Dentro de la función extractFeatures
    private float[] extractFeatures(float[] x, float[] y, float[] time, float[] pressure) {
        int numFeatures = 25; // Ajusta el tamaño según las características
        float derivscale = 2;
        float mp = (float) 0.1;
        float[] features = new float[numFeatures];


        // Convert float[] arrays to Float[] arrays
        Float[] xFloat = new Float[x.length];
        for (int i = 0; i < x.length; i++) {
            xFloat[i] = x[i];
        }

        Float[] yFloat = new Float[y.length];
        for (int i = 0; i < y.length; i++) {
            yFloat[i] = y[i];
        }

        // Calculate maximum and minimum values for x and y
        float xMax = Float.MIN_VALUE;
        float xMin = Float.MAX_VALUE;
        float yMax = Float.MIN_VALUE;
        float yMin = Float.MAX_VALUE;

        for (int i = 0; i < xFloat.length; i++) {
            if (xFloat[i] > xMax) {
                xMax = xFloat[i];
            }
            if (xFloat[i] < xMin) {
                xMin = xFloat[i];
            }
        }

        for (int i = 0; i < yFloat.length; i++) {
            if (yFloat[i] > yMax) {
                yMax = yFloat[i];
            }
            if (yFloat[i] < yMin) {
                yMin = yFloat[i];
            }
        }

        // Aspect ratio
        float aspectRatio = (yMax - yMin) / (xMax - xMin);
        features[0] = aspectRatio;

        // Desviación media de X e Y
        features[1] = calculateMeanAndMad(floatArrayToDoubleArray(x))[1];
        features[2] = calculateMeanAndMad(floatArrayToDoubleArray(y))[1];

        // Media y desviación estándar de dX/dT y dY/dT
        double[] dXdT = calculateDerivative(x, time);
        double[] dYdT = calculateDerivative(y, time);
        float[] maddXdT = calculateMeanAndMad(dXdT);
        float[] maddYdT = calculateMeanAndMad(dYdT);
        features[3] = maddXdT[0];
        features[4] = maddXdT[1];
        features[5] = maddYdT[0];
        features[6] = maddYdT[1];

        // Correlaciones

        // Convert float[] arrays to double[] arrays using the utility function
        double[] xDouble = floatArrayToDoubleArray(x);
        double[] yDouble = floatArrayToDoubleArray(y);
        double[] pressureDouble = floatArrayToDoubleArray(pressure);

        // Now you can pass the double[] arrays to calculateCorrelations
        int index = 6;
        double[] correlations = calculateCorrelations(xDouble, yDouble, pressureDouble, dXdT, dYdT);
        for (double correlation : correlations) {
            features[index++] = (float) correlation;
        }

        // Número de mínimos locales
        int numXMinima = countLocalMinima(floatArrayToDoubleArray(x), mp);
        int numYMinima = countLocalMinima(floatArrayToDoubleArray(y), mp);
        int numXdTMinima = countLocalMinima(dXdT, derivscale * mp);
        int numYdTMinima = countLocalMinima(dYdT, derivscale * mp);

        // Número de máximos locales
        int numXMaxima = countLocalMaxima(floatArrayToDoubleArray(x), mp);
        int numYMaxima = countLocalMaxima(floatArrayToDoubleArray(y), mp);
        int numXdTMaxima = countLocalMaxima(dXdT, derivscale * mp);
        int numYdTMaxima = countLocalMaxima(dYdT, derivscale * mp);

        features[17] = numXMinima;
        features[18] = numXMaxima;
        features[19] = numYMinima;
        features[20] = numYMaxima;
        features[21] = numXdTMinima;
        features[22] = numXdTMaxima;
        features[23] = numYdTMinima;
        features[24] = numYdTMaxima;

        return features;
    }


    private MappedByteBuffer loadModelFile(String modelFileName) throws IOException {
        AssetManager assetManager = getAssets();
        InputStream inputStream = assetManager.open(modelFileName);
        byte[] modelData = new byte[inputStream.available()];
        inputStream.read(modelData);
        inputStream.close();
        return (MappedByteBuffer) ByteBuffer.allocateDirect(modelData.length)
                .order(ByteOrder.nativeOrder())
                .put(modelData);
    }

*/
/*
    public static int[] getTopNClasses(float[] probabilities, int n) {
        if (probabilities == null || probabilities.length == 0 || n <= 0 || n > probabilities.length) {
            throw new IllegalArgumentException("Invalid input values.");
        }

        // Create a priority queue to keep track of the top N classes by probability
        PriorityQueue<IndexProbabilityPair> queue = new PriorityQueue<>(n);

        for (int i = 0; i < probabilities.length; i++) {
            float probability = probabilities[i];

            // Create a pair of index and probability
            IndexProbabilityPair pair = new IndexProbabilityPair(i, probability);

            // Add the pair to the priority queue
            queue.offer(pair);

            // If the queue size exceeds N, remove the lowest probability pair
            if (queue.size() > n) {
                queue.poll();
            }
        }

        // Extract the indices of the top N classes from the priority queue
        int[] topClasses = new int[n];
        int index = n - 1;

        while (!queue.isEmpty()) {
            topClasses[index--] = queue.poll().index;
        }

        return topClasses;
    }
*/
    /*
    // Helper class to store index and probability pairs
    private static class IndexProbabilityPair implements Comparable<IndexProbabilityPair> {
        private final int index;
        private final float probability;

        public IndexProbabilityPair(int index, float probability) {
            this.index = index;
            this.probability = probability;
        }

        @Override
        public int compareTo(IndexProbabilityPair other) {
            // Compare by probability in reverse order (highest probability first)
            return Float.compare(other.probability, this.probability);
        }
    }
*/

    /*
    private void predict(float[] inputArray){ //Bitmap bm
        try {
            Log.d("TAG", "tflite inputArraySize: " + inputArray.length);
            for (float inTF:inputArray){
                Log.d("TAG", "tflite inputArray: " + inTF);

            }
            //tflite.run(inputArray, outputValues);
            tflite.run(inputArray, probabilityBuffer.getBuffer());
            // Post-processor which dequantize the result
            //TensorProcessor probabilityProcessor =
            //        new TensorProcessor.Builder().add(new DequantizeOp(0, 1/255)).build();
            //TensorBuffer dequantizedBuffer = probabilityProcessor.process(probabilityBuffer);



            float[] outputArray = probabilityBuffer.getFloatArray();
            Log.d("TAG", "tflite outputArraySize: " + outputArray.length);
            for (float outTF:outputArray){
                Log.d("TAG", "tflite outputArray: " + outTF);

            }
            //float[] outputArray = outputValues[0];
            //float confidence = outputArray[0];

            Toast toast = Toast.makeText(this, "tflite prediction: "+ outputArray[0] +" confidance: ", Toast.LENGTH_SHORT);
            TextView v = (TextView) toast.getView().findViewById(android.R.id.message);
            if( v != null) v.setGravity(Gravity.CENTER);
            toast.show();
            return;
        }catch (RuntimeException e){
            // Handle the exception or log it for debugging
            Log.d("TAG", "tflite error 2: "+e.getMessage());
            e.printStackTrace();
        }

        ////
        try {
            // Inicializar el arreglo de salida
            outputValues = new float[343][3];

            Log.d("TAG", "bitmapim size: "+bm.getHeight() +" "+ bm.getWidth());

            // Assuming a batch size of 1
            int inputWidth = 227;
            int inputHeight = 400;
            int channelCount = 3;  // Assuming RGB image

            float[][][][] inputArray = new float[1][inputWidth][inputHeight][channelCount];

            Log.d("TAG", "tflite in size: "+inputArray.length);

            for (int x = 0; x < inputWidth; x++) {
                //Log.d("TAG", "tflite x: "+x);
                for (int y = 0; y < inputHeight; y++) {
                    int pixel = bm.getPixel(x, y);
                    //Log.d("TAG", "tflite x y, px: "+x+","+y+" - "+ ((pixel >> 16) & 0xFF) / 255.0f+","+((pixel >> 8) & 0xFF) / 255.0f+","+(pixel & 0xFF) / 255.0f);
                    //Log.d("TAG", "tflite px: "+pixel);
                    // Extract and normalize RGB values
                    inputArray[0][x][y][0] = ((pixel >> 16) & 0xFF) / 255.0f;
                    inputArray[0][x][y][1] = ((pixel >> 8) & 0xFF) / 255.0f;
                    inputArray[0][x][y][2] = (pixel & 0xFF) / 255.0f;
                }
            }



            //Log.d("TAG", "preprocessdata: "+inputArray.length);

            // Realizar predicción utilizando el modelo TensorFlow Lite
            tflite.run(inputArray, outputValues);

            //Log.d("TAG","tflite numClasses 0: "+outputValues[0].length);

            // Assuming 'outputArray' contains the model's output
            float[] outputArray = outputValues[0]; // Assuming a single inference
            Log.d("TAG","tflite output 1: "+outputValues[0][0]);

            // Find the top N predicted classes
            int numTopClasses = 3; // Example: Get the top 3 predicted classes
            int[] topClasses = getTopNClasses(outputArray, numTopClasses);

            int predictedClassIndex = 0;
            float confidence = 0f;
            // Display the top predicted classes
            for (int i = 0; i < numTopClasses; i++) {
                predictedClassIndex = topClasses[i];
                confidence = outputArray[predictedClassIndex];
                // You can now use 'predictedClassIndex' and 'confidence' as needed.
            }

            Toast toast = Toast.makeText(this, "tflite prediction: "+ predictedClassIndex +" confidance: "+confidence, Toast.LENGTH_SHORT);
            TextView v = (TextView) toast.getView().findViewById(android.R.id.message);
            if( v != null) v.setGravity(Gravity.CENTER);
            toast.show();
            return;
        }

        catch (Exception e) {
            // Handle the exception or log it for debugging
            Log.d("TAG", "tflite error 2: "+e.getMessage());
            e.printStackTrace();
        }
    } */



}


/*
        // Define the size of the arrays
        int arraySize = timeOffsets.size();

        // Create float arrays for coordinates (x and y)
        float[] xCoordinates = new float[arraySize];
        float[] yCoordinates = new float[arraySize];

        // Initialize the arrays with your coordinate values
        // For example, if you have two ArrayLists for x and y coordinates:
        for (int i = 0; i < arraySize; i++) {
            xCoordinates[i] = coordinates.get(i).first;
            yCoordinates[i] = coordinates.get(i).second;
        }

        // Create an array for timeOffsets (assuming you have timeOffsets as well)
        float[] timeOffsetsF = new float[arraySize];

        for (int i = 0; i < timeOffsets.size(); i++) {
            timeOffsetsF[i] = timeOffsets.get(i).floatValue();
        }

        // Create an array for pressures filled with 90.0
        float[] pressures = new float[arraySize];
        Arrays.fill(pressures, 1.0f);

        // Call the extractFeatures function with the arrays
        //float[] preprocessedData = extractFeatures(xCoordinates, yCoordinates, timeOffsetsF, pressures);
        //Log.d("TAG", "preprocessedData error: " + preprocessedData[0]+" "+ preprocessedData[1]+" "+ preprocessedData[2]+" "+ preprocessedData[3]+" "+ preprocessedData[4]+" " + preprocessedData[5]+" "+ preprocessedData[6]+" "+ preprocessedData[7]+" "+ preprocessedData[8]+" "+ preprocessedData[9] +" "+ preprocessedData[10]+" "+ preprocessedData[11]+" "+ preprocessedData[12]+" "+" "+ preprocessedData[13]+" "+ preprocessedData[14]+" "+ preprocessedData[15]+" "+ preprocessedData[16]+" "+ preprocessedData[17]+" "+ preprocessedData[18]+" "+ preprocessedData[19]+" "+ preprocessedData[20]+" "+ preprocessedData[21]+" "+ preprocessedData[22]+" "+ preprocessedData[23]+" "+ preprocessedData[24]);
        */