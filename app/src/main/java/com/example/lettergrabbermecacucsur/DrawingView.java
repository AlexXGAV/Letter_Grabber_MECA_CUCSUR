package com.example.lettergrabbermecacucsur;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.util.Pair;
import android.widget.ImageView;


import java.util.ArrayList;
import java.util.List;

/**
 * TODO: document your custom view class.
 */
public class DrawingView extends ImageView {

    private Paint paint;
    private List<Pair<Float, Float>> coordinates;

    public DrawingView(Context context) {
        super(context);
        init();
    }

    public DrawingView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        paint = new Paint();
        //paint.setColor(Color.BLACK);
        paint.setStyle(Paint.Style.FILL);
        paint.setStrokeWidth(5f);
        coordinates = new ArrayList<>();
    }


    public void addPoint(float x, float y) {
        coordinates.add(new Pair<>(x, y));
        invalidate(); // Redibujar el punto recién agregado
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        paint.setStyle(Paint.Style.FILL);

        for (Pair<Float, Float> coordinate : coordinates) {
            canvas.drawPoint(coordinate.first, coordinate.second, paint);
        }
    }


    public void clearDrawing() {
        coordinates.clear();
        invalidate(); // Redraw the view to clear the canvas
    }

}
