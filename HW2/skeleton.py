#################################
# Your name: Nathan Bloch
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        # TODO: Implement me
        samples = np.zeros((m, 2))
        xs = np.random.uniform(0, 1, m)
        xs = np.sort(xs)
        for i in range(0, m):
            x = xs[i]
            samples[i][0] = x
            y_rand = np.random.uniform(0, 1, 1)
            if 0 <= x <= 0.2 or 0.4 <= x <= 0.6 or 0.8 <= x <= 1:
                if y_rand <= 0.8:
                    samples[i][1] = 1
                else:
                    samples[i][1] = 0
            else:
                if y_rand <= 0.1:
                    samples[i][1] = 1
                else:
                    samples[i][1] = 0
        return samples



    def draw_sample_intervals(self, m, k):
        """
        Plots the data as asked in (a) i ii and iii.
        Input: m - an integer, the size of the data sample.
               k - an integer, the maximum number of intervals.

        Returns: None.
        """
        # TODO: Implement me
        samples = self.sample_from_D(m)
        best_intervals = intervals.find_best_interval(samples[:, 0], samples[:, 1], k)
        plt.plot(samples[:, 0], samples[:, 1], 'bo', marker='.')
        plt.xlabel("X : values")
        plt.ylabel("Y : labels")
        plt.ylim((-0.1, 1.1))
        plt.xlim((0, 1))
        plt.vlines([0.2, 0.4, 0.6, 0.8], ymin=-0.1, ymax=1.1, linestyles='dashed', color="black")
        for interval in best_intervals[0]:
            plt.hlines(y=-0.1, xmin=interval[0], xmax=interval[1], color="red", linestyles='solid', linewidth=8)
        # plt.show()


    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        # TODO: Implement the loop
        index = 0
        results = np.zeros((len(range(m_first, m_last+1, step)), 2))
        for m in range(m_first, m_last+1, step):
            cnt_true_error = 0
            cnt_emp_error = 0
            for i in range(T):
                samples = self.sample_from_D(m)
                best_intervals = intervals.find_best_interval(samples[:, 0], samples[:, 1], k)
                cnt_emp_error += best_intervals[1] / m
                cnt_true_error += self.calc_true_error(best_intervals[0])

            results[index][0] = cnt_emp_error / T
            results[index][1] = cnt_true_error / T
            index += 1

        # self.plotC(m_first, m_last, step, results)
        return results

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        # TODO: Implement the loop
        samples = self.sample_from_D(m)
        index = 0
        best_k = 0
        results = np.zeros((len(range(k_first, k_last + 1, step)), 2))
        for k in range(k_first, k_last + 1, step):
            erm_result = intervals.find_best_interval(samples[:, 0], samples[:, 1], k)
            results[index][0] = erm_result[1] / m
            results[index][1] = self.calc_true_error(erm_result[0])
            if (results[index][0] < results[best_k][0]):
                best_k = index
            index += 1

        # self.plotD(k_first, k_last, step, results)
        return best_k + 1


    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Runs the experiment in (d).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        # TODO: Implement the loop
        samples = self.sample_from_D(m)
        index = 0
        best_k = 0
        results = np.zeros((len(range(k_first, k_last + 1, step)), 3))
        for k in range(k_first, k_last + 1, step):
            erm_result = intervals.find_best_interval(samples[:, 0], samples[:, 1], k)
            results[index][0] = erm_result[1] / m
            results[index][1] = self.calc_true_error(erm_result[0])
            results[index][2] = self.calc_penalty(m, k)
            if(results[index][0] + results[index][2] < results[best_k][0] + results[best_k][2]):
                best_k = index
            index += 1
        # self.plotE(k_first, k_last, step, results)
        return best_k + 1

    def cross_validation(self, m, T):
        """Finds a k that gives a good test error.
        Chooses the best hypothesis based on 3 experiments.
        Input: m - an integer, the size of the data sample.
               T - an integer, the number of times the experiment is performed.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        # TODO: Implement me
        samples = self.sample_from_D(m)
        hold_out_size = 0.2
        hold_out = np.zeros((int(hold_out_size * m), 2))
        s1 = np.zeros((int((1 - hold_out_size) * m), 2))
        s1_index, s2_index = 0, 0
        for i in range(m):
            if i % 5 == 0:
                hold_out[s2_index][0] = samples[i][0]
                hold_out[s2_index][1] = samples[i][1]
                s2_index += 1
            else:
                s1[s1_index][0] = samples[i][0]
                s1[s1_index][1] = samples[i][1]
                s1_index += 1

        results = list()
        for i in range(T):
            empiric_results = list()
            erm_results = list()
            for k in range(1, 11):
                erm_result = intervals.find_best_interval(s1[:, 0], s1[:, 1], k)
                erm_results.append(erm_result[0])
                empiric_results.append(self.calc_emp_error(hold_out, erm_result[0]))

            minimal_k = empiric_results.index(min(empiric_results))
            results.append((minimal_k + 1, empiric_results[minimal_k], erm_results[minimal_k]))

        best = 0
        for i in range(1, T):
            if results[i][1] < results[best][1]:
                best = i

        return results[best][0]

    #################################
    # Place for additional methods

    def calc_emp_error(self, samples, intervals):
        count = 0
        errors = 0
        for sample in samples:
            h = 0
            for interval in intervals:
                if interval[0] <= sample[0] <= interval[1]:
                    h = 1
            if h != sample[1]:
                errors += 1
            count += 1

        return errors / count


    def calc_true_error(self, best_intervals):
        """Calculates the true error of the intervals given.
           Returns an integer type.
        """
        sum1 = 0        # represents the sum of intervals where h(x) =1 (based on best_intervals)
                        # and x in [0,0.2] union [0.4,0.6] union [0.8,1].
        sum2 = 0        # represents the sum of intervals where h(x) =1 (based on best_intervals)
                        # and x in [0.2,0.4] union [0.6,0.8].
        true_error = 0
        for i in range(5):
            for interval in best_intervals:
                if(i % 2 == 0):
                    sum1 += max(0, min(0.2*i + 0.2, interval[1]) - max(0.2*i, interval[0]))
                else:
                    sum2 += max(0, min(0.2*i + 0.2, interval[1]) - max(0.2*i, interval[0]))

        # This section is based on the mathematics in the pdf submitted. Main idea: calc the integration wisely.
        # Integrate over [0,0.2] union [0.4, 0.6] union [0.8, 1].
        true_error += 0.2 * sum1                # where h(x) = 1
        true_error += 0.8 * (0.6 - sum1)        # where h(x) = 0, which is the complement.

        # Integrate over [0.2,0.4] union [0.6,0.8].
        true_error += 0.9 * sum2                # where h(x) = 1
        true_error += 0.1 * (0.4 - sum2)        # where h(x) = 0, which is the complement.
        return true_error

    def calc_penalty(self, n, k):
        penalty = 0
        penalty += 2 * k * np.log((np.e * n) / k)
        penalty += np.log(4 / 0.1)
        penalty *= (8 / n)
        return penalty ** 0.5


    def plotC(self, m_first, m_last, step, results):
        x_labels = [m for m in range(m_first, m_last+1, step)]
        emp_line, = plt.plot(x_labels, results[:, 0], 'ro', color="red")
        true_line, = plt.plot(x_labels, results[:, 1], 'bo', color="black")

        plt.ylabel("Average Error Values")
        plt.xlabel("M Values")
        plt.xlim((x_labels[0] - 5, x_labels[-1] + 5))
        plt.legend((emp_line, true_line), ('Empirical Error', 'True Error'))
        # plt.show()

    def plotD(self, k_first, k_last, step, results):
        x_labels = [k for k in range(k_first, k_last+1, step)]
        emp_line, = plt.plot(x_labels, results[:, 0], color="red")
        true_line, = plt.plot(x_labels, results[:, 1], color="black")

        plt.ylabel("Error Values")
        plt.xlabel("K Values")
        plt.xlim((x_labels[0] - 1, x_labels[-1] + 1))
        plt.xticks(np.arange(0, 12, step=1))
        plt.legend((emp_line, true_line), ('Empirical Error', 'True Error'))
        # plt.show()

    def plotE(self, k_first, k_last, step, results):
        x_labels = [k for k in range(k_first, k_last+1, step)]
        emp_line, = plt.plot(x_labels, results[:, 0], color="red")
        true_line, = plt.plot(x_labels, results[:, 1], color="black")
        penalty_line, = plt.plot(x_labels, results[:, 2], color="blue")
        sum = np.array(results[:, 0]) + np.array(results[:, 2])
        sum_line, = plt.plot(x_labels, sum, color="green")
        plt.ylabel("Error Values")
        plt.xlabel("K Values")
        plt.xlim((x_labels[0], x_labels[-1]))
        plt.xticks(np.arange(0, 11, step=1))
        plt.legend((emp_line, true_line, penalty_line, sum_line), ('Empirical Error', 'True Error', 'Penalty', 'Sum'))
        # plt.show()

    #################################


if __name__ == '__main__':
    ass = Assignment2()
    ass.draw_sample_intervals(100, 3)
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500, 3)

