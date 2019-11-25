package main.java.mdp;

import burlap.behavior.policy.EpsilonGreedy;
import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.domain.singleagent.blockdude.BlockDude;
import burlap.domain.singleagent.blockdude.BlockDudeLevelConstructor;
import burlap.domain.singleagent.blockdude.BlockDudeTF;
import burlap.domain.singleagent.blockdude.BlockDudeVisualizer;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.mdp.auxiliary.stateconditiontest.TFGoalCondition;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.common.GoalBasedRF;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.FactoredModel;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.statehashing.HashableStateFactory;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.Visualizer;

import java.util.List;

public class BlockDudeExperiment {
    private final int maxx = 24;
    private final int maxy = 24;
    SADomain domain;
    BlockDude bd;
    TerminalFunction tf;
    RewardFunction rf;
    TFGoalCondition goalCondition;
    State initialState;
    HashableStateFactory hashingFactory;
    SimulatedEnvironment env;

    public BlockDudeExperiment(double goalReward, double defaultReward, int level) {
        bd = new BlockDude(maxx, maxy);
        domain = bd.generateDomain();
        tf = new BlockDudeTF();
        goalCondition = new TFGoalCondition(tf);
        rf = new GoalBasedRF(goalCondition, goalReward, defaultReward);
//        rf = new UniformCostRF();
        hashingFactory = new SimpleHashableStateFactory();

        if (level == 1) {
            initialState = BlockDudeLevelConstructor.getLevel1(domain);
        } else if (level == 2) {
            initialState = BlockDudeLevelConstructor.getLevel2(domain);
        } else {
            initialState = BlockDudeLevelConstructor.getLevel3(domain);
        }
        env = new SimulatedEnvironment(domain, initialState);

    }
    public void simpleValueFunctionVis(ValueFunction valueFunction, Policy p){

        List<State> allStates = StateReachability.getReachableStates(
                initialState, domain, hashingFactory);
        ValueFunctionVisualizerGUI gui = GridWorldDomain.getGridWorldValueFunctionVisualization(
                allStates, maxx, maxy, valueFunction, p);
        gui.initGUI();

    }
    public void visualize(String outputpath){
        Visualizer v = BlockDudeVisualizer.getVisualizer(bd.getMaxx(),bd.getMaxy());
        new EpisodeSequenceVisualizer(v, domain, outputpath);
    }

    public void valueIterationExample(String outputPath, double gamma){

        Planner planner = new ValueIteration(domain, gamma, hashingFactory, 0.001, 100);
        Policy p = planner.planFromState(initialState);

        PolicyUtils.rollout(p, initialState, domain.getModel()).write(outputPath + String.format("bde_vi_gam%s", gamma));

        simpleValueFunctionVis((ValueFunction)planner, p);

    }

    public void policyIterationExample(String outputPath, double gamma){

        Planner planner = new PolicyIteration(domain, gamma, hashingFactory, 0.001, 100, 100, 100);
        Policy p = planner.planFromState(initialState);

        PolicyUtils.rollout(p, initialState, domain.getModel()).write(outputPath + String.format("bde_pi_gam%s", gamma));

        simpleValueFunctionVis((ValueFunction)planner, p);

    }

    public void qLearningExample(String outputPath, double gamma, double qInit, double learningRate, double epsilon, int numEpisodes){

        QLearning agent = new QLearning(domain, gamma, hashingFactory, qInit, learningRate);
        agent.setLearningPolicy(new EpsilonGreedy(agent, epsilon));
        //run learning for n episodes
        for(int i = 0; i < numEpisodes; i++){
            Episode e = agent.runLearningEpisode(env, 10000);

            e.write(outputPath + "ql_" + i);
            System.out.println(i + ": " + e.maxTimeStep());

            //reset environment for next learning episode
            env.resetEnvironment();
        }

    }

    public void experimentAndPlotter(double gamma, double qInit, double learningRate, double epsilon, int numEpisodes, int level){

        //different reward function for more structured performance plots
        ((FactoredModel)domain.getModel()).setRf(new GoalBasedRF(this.goalCondition, 5.0, -0.1));

        LearningAgentFactory qLearningFactory = new LearningAgentFactory() {

            public String getAgentName() {
                return "Q-Learning";
            }

            public LearningAgent generateAgent() {
                QLearning agent = new QLearning(domain, gamma, hashingFactory, qInit, learningRate);
                agent.setLearningPolicy(new EpsilonGreedy(agent, epsilon));
                return agent;
            }
        };

        LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter(
                env, 10, 10000, qLearningFactory);
        exp.setUpPlottingConfiguration(400, 400, 2, 1000,
                TrialMode.MOST_RECENT_AND_AVERAGE,
                PerformanceMetric.STEPS_PER_EPISODE,
                PerformanceMetric.AVERAGE_EPISODE_REWARD);

        long start = System.currentTimeMillis();
        exp.startExperiment();
        System.out.println(System.currentTimeMillis()-start);
        String filepath = String.format("bde_qlearning%d_%sdf_%sepsilon_%slr_", level, gamma, epsilon, learningRate);
        exp.writeStepAndEpisodeDataToCSV(filepath);

    }

    public static void main(String[] args) {
        int level = 1;
        String algorithm = "QL";
        double gamma = 0.99;
        double qInit = 0.01;
        double learningRate = 0.8;
        double epsilon = 0.01;
        int numEpisodes = 50;

        BlockDudeExperiment bde = new BlockDudeExperiment(5., -0.1, level);
        String outputPath = "output/";

        long start = System.currentTimeMillis();

        if (algorithm == "PI") {
            bde.policyIterationExample(outputPath, gamma);
        } else if (algorithm == "VI") {
            bde.valueIterationExample(outputPath, gamma);
        } else if (algorithm == "QL") {
            bde.qLearningExample(outputPath, gamma, qInit, learningRate, epsilon, numEpisodes);
        }
        long wallClock = System.currentTimeMillis() - start;
        System.out.printf("Wall clock for %s: %d\n", algorithm, wallClock);
        if (algorithm == "QLE") {
            bde.experimentAndPlotter(gamma, qInit, learningRate, epsilon, numEpisodes, level);
            System.out.printf("bde_%s_%sepsilon_%sLR_%d#eps_%dms", algorithm, epsilon, learningRate, numEpisodes, wallClock);
        } else {
            System.out.printf("bde_%s_%sdf_%dms", algorithm, gamma, wallClock);
        }
        bde.visualize(outputPath);
    }
}