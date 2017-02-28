(ns deep-learning.course1
  (:gen-class))

(def training-dataset
  (array-map
    [1 1] 1
    [0 0] 0
    [1 0] 0
    [0 1] 0))

(defn step-activator [x] (if (> x 0) 1 0))

(defn predict [activator xs wx bias] (activator (+ bias (apply + (map * xs wx)))))

(def step-predict (partial predict step-activator))

(defn calc-weight [xs wx rate]
  (map #(+ %2 (* rate %1)) xs wx))

(defn train [dataset iteration rate]
  (loop [i iteration
         samples dataset
         wx [0 0]
         bias 0]
    (if (> i 0)
      (if (first samples)
        (let [[xs label] (first samples)
              output (step-predict xs wx bias)
              delta (- label output)
              weights (calc-weight xs wx (* rate delta))]
          (recur i (rest samples) weights (+ bias (* rate delta))))
        (recur (dec i) dataset wx bias))
      [wx bias])))

(defn main []
  (let [[wx bias] (train training-dataset 10 0.1)
        predict-for-this #(step-predict % wx bias)]
    (println "1 and 1 = " (predict-for-this [1 1]))
    (println "0 and 0 = " (predict-for-this [0 0]))
    (println "1 and 0 = " (predict-for-this [1 0]))
    (println "0 and 1 = " (predict-for-this [0 1]))))
