"""Small synthetic quadtree ownership, ranking and reproducibility tests."""
import json
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.spatial.quadtree import build_quadtree, select_quadtree_matches, tree_statistics
from src.evaluation.visualization import plot_quadtree
from scripts.tycho_quadtree import load_m11, run


class QuadtreeTests(unittest.TestCase):
    def setUp(self):
        self.points = np.array([[1., 1.], [5., 1.], [1., 5.], [5., 5.]])

    def tearDown(self):
        plt.close('all')

    def test_empty_tree(self):
        tree = build_quadtree(np.empty((0, 2)), (8, 8))
        self.assertTrue(tree.root.is_leaf)
        stats = tree_statistics(tree)
        self.assertEqual(stats['total_node_count'], 1)
        self.assertEqual(stats['leaf_count'], 1)
        self.assertEqual(stats['occupied_leaf_count'], 0)
        self.assertEqual(stats['maximum_reached_depth'], 0)
        self.assertIsNone(stats['mean_points_per_occupied_leaf'])
        result = select_quadtree_matches(tree, np.empty((0, 2)), confidence=[], residuals=[])
        self.assertEqual(result.source.shape, (0, 2))
        self.assertEqual(result.original_indices.size, 0)
        json.dumps(stats, allow_nan=False)

    def test_single_point(self):
        tree = build_quadtree([[1, 1]], (8, 8), capacity=1)
        self.assertEqual(tree.root.original_indices, (0,))
        self.assertTrue(tree.root.is_leaf)
        self.assertEqual(tree_statistics(tree)['mean_points_per_occupied_leaf'], 1)

    def test_no_subdivision_at_capacity(self):
        tree = build_quadtree(self.points, (8, 8), capacity=4)
        self.assertEqual(len(tree.nodes), 1)

    def test_subdivision_above_capacity(self):
        tree = build_quadtree(self.points, (8, 8), capacity=3)
        self.assertFalse(tree.root.is_leaf)
        self.assertEqual(len(tree.root.children), 4)
        self.assertEqual(len(tree.leaves), 4)

    def test_maximum_depth(self):
        points = np.tile([1., 1.], (30, 1))
        tree = build_quadtree(points, (32, 32), capacity=1, max_depth=2)
        self.assertEqual(tree_statistics(tree)['maximum_reached_depth'], 2)
        self.assertEqual(max(len(n.original_indices) for n in tree.leaves), 30)
        self.assertEqual(len(build_quadtree(points, (32, 32), max_depth=0).nodes), 1)

    def test_split_boundaries(self):
        xy = [[np.nextafter(4., 0), np.nextafter(4., 0)], [4, 0], [0, 4], [4, 4]]
        tree = build_quadtree(xy, (8, 8), capacity=1, max_depth=1)
        self.assertEqual([n.original_indices for n in tree.root.children], [(0,), (1,), (2,), (3,)])

    def test_final_valid_pixel(self):
        tree = build_quadtree([[0, 0], [320, 312], [np.nextafter(321., 0), np.nextafter(313., 0)]],
                              (313, 321), capacity=1)
        last = next(n for n in tree.leaves if 2 in n.original_indices)
        self.assertIn(1, last.original_indices)
        self.assertEqual(last.bounds[2:], (321., 313.))

    def test_negative_coordinate(self):
        with self.assertRaises(ValueError):
            build_quadtree([[-.001, 0]], (8, 8))

    def test_outside_coordinate(self):
        for point in ([8, 0], [0, 8], [9, 9]):
            with self.assertRaises(ValueError):
                build_quadtree([point], (8, 8))

    def test_nonfinite_coordinates(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.assertRaises(ValueError):
                build_quadtree([[value, 0]], (8, 8))

    def test_odd_dimensions(self):
        tree = build_quadtree([[0, 0], [2.5, 0], [0, 3.5], [4, 6]], (7, 5), capacity=1)
        self.assertEqual(tree.root.children[0].bounds, (0., 0., 2.5, 3.5))
        self.assertEqual(tree.root.children[-1].bounds, (2.5, 3.5, 5., 7.))
        self.assertEqual([c.original_indices for c in tree.root.children], [(0,), (1,), (2,), (3,)])

    def test_duplicate_coordinates(self):
        points = np.tile([1.25, 1.25], (100, 1))
        tree = build_quadtree(points, (8, 8), capacity=1, max_depth=4)
        occupied = [leaf for leaf in tree.leaves if leaf.original_indices]
        self.assertEqual(len(occupied), 1)
        self.assertEqual(len(occupied[0].original_indices), 100)
        self.assertLessEqual(tree_statistics(tree)['maximum_reached_depth'], 4)

    def test_no_point_loss(self):
        points = np.random.default_rng(42).uniform(0, 31, (200, 2))
        tree = build_quadtree(points, (31, 31), capacity=3)
        all_indices = [i for leaf in tree.leaves for i in leaf.original_indices]
        self.assertEqual(sorted(all_indices), list(range(len(points))))
        for node in tree.nodes:
            if not node.is_leaf:
                self.assertEqual(sorted(i for c in node.children for i in c.original_indices), list(node.original_indices))

    def test_no_point_duplication_and_containment(self):
        points = np.array([(x, y) for x in range(8) for y in range(8)])
        tree = build_quadtree(points, (8, 8), capacity=1)
        indices = [i for leaf in tree.leaves for i in leaf.original_indices]
        self.assertEqual(len(indices), len(set(indices)))
        for leaf in tree.leaves:
            x0, y0, x1, y1 = leaf.bounds
            xy = points[list(leaf.original_indices)]
            self.assertTrue(((xy >= [x0, y0]) & (xy < [x1, y1])).all())

    def test_child_order_and_preorder_ids(self):
        tree = build_quadtree(self.points, (8, 8), capacity=1)
        self.assertEqual([n.node_id for n in tree.nodes], list(range(5)))
        self.assertEqual([n.bounds for n in tree.leaves],
                         [(0, 0, 4, 4), (4, 0, 8, 4), (0, 4, 4, 8), (4, 4, 8, 8)])

    def test_deterministic_selection_without_scores(self):
        points = np.repeat(self.points, 3, axis=0)
        tree = build_quadtree(points, (8, 8), capacity=3)
        selected = select_quadtree_matches(tree, points)
        np.testing.assert_array_equal(selected.original_indices, [0, 3, 6, 9])

    def test_residual_ordering(self):
        tree = build_quadtree(self.points, (8, 8), capacity=4)
        selected = select_quadtree_matches(tree, self.points, residuals=[3, 2, 1, 0])
        np.testing.assert_array_equal(selected.original_indices, [3])

    def test_confidence_ordering(self):
        tree = build_quadtree(self.points, (8, 8), capacity=4)
        selected = select_quadtree_matches(tree, self.points, confidence=[.2, .9, .8, .7])
        np.testing.assert_array_equal(selected.original_indices, [1])

    def test_lexicographic_ties(self):
        tree = build_quadtree(self.points, (8, 8), capacity=4)
        selected = select_quadtree_matches(tree, self.points, max_per_leaf=2,
                                           residuals=[1, 1, .5, 1], confidence=[.8, .9, .1, .9])
        np.testing.assert_array_equal(selected.original_indices, [1, 2])
        selected = select_quadtree_matches(tree, self.points, residuals=[1]*4, confidence=[.5]*4)
        np.testing.assert_array_equal(selected.original_indices, [0])

    def test_index_preservation_and_independent_copies(self):
        destination = self.points.copy()
        tree = build_quadtree(destination, (8, 8), capacity=4)
        source = destination+10
        confidence, residuals = np.array([.1, .2, .3, .4]), np.array([4, 3, 2, 1])
        selected = select_quadtree_matches(tree, source, max_per_leaf=2,
                                           confidence=confidence, residuals=residuals, source_shape=(20, 20))
        np.testing.assert_array_equal(selected.original_indices, [2, 3])
        np.testing.assert_array_equal(selected.source, source[[2, 3]])
        np.testing.assert_array_equal(selected.destination, destination[[2, 3]])
        np.testing.assert_array_equal(selected.leaf_ids, [0, 0])
        np.testing.assert_array_equal(selected.confidence, confidence[[2, 3]])
        np.testing.assert_array_equal(selected.residuals, residuals[[2, 3]])
        destination[:] = 7
        np.testing.assert_array_equal(tree.destination, self.points)
        self.assertFalse(tree.destination.flags.writeable)
        selected.source[:] = 99
        np.testing.assert_array_equal(source, self.points+10)

    def test_leaf_cap_distinct_from_capacity(self):
        points = np.repeat(self.points, 10, axis=0)
        tree = build_quadtree(points, (8, 8), capacity=1, max_depth=1)
        selected = select_quadtree_matches(tree, points, max_per_leaf=2)
        self.assertEqual(len(selected.source), 8)
        self.assertTrue(all(np.count_nonzero(selected.leaf_ids == leaf.node_id) == 2 for leaf in tree.leaves))

    def test_fair_global_allocation(self):
        points = np.repeat(self.points, 4, axis=0)
        tree = build_quadtree(points, (8, 8), capacity=4)
        selected = select_quadtree_matches(tree, points, max_per_leaf=4, max_matches=6)
        np.testing.assert_array_equal(selected.original_indices, [0, 1, 4, 5, 8, 12])
        np.testing.assert_array_equal(np.unique(selected.leaf_ids, return_counts=True)[1], [2, 2, 1, 1])
        small = select_quadtree_matches(tree, points, max_per_leaf=4, max_matches=2)
        np.testing.assert_array_equal(small.original_indices, [0, 4])

    def test_tree_statistics(self):
        tree = build_quadtree([[0, 0], [1, 1], [5, 1], [5, 5]], (8, 8), capacity=1, max_depth=1)
        stats = tree_statistics(tree)
        self.assertEqual(stats['total_node_count'], 5)
        self.assertEqual(stats['leaf_count'], 4)
        self.assertEqual(stats['occupied_leaf_count'], 3)
        self.assertEqual(stats['minimum_leaf_depth'], 1)
        self.assertEqual(stats['mean_occupied_leaf_depth'], 1)
        self.assertEqual(stats['maximum_points_per_occupied_leaf'], 2)
        self.assertEqual(stats['minimum_points_per_occupied_leaf'], 1)
        self.assertAlmostEqual(stats['mean_points_per_occupied_leaf'], 4/3)
        self.assertAlmostEqual(stats['std_points_per_occupied_leaf'], np.std([2, 1, 1]))
        self.assertEqual(stats['occupied_leaves_by_depth'], [{'depth': 0, 'count': 0}, {'depth': 1, 'count': 3}])

    def test_repeated_call_reproducibility(self):
        points = np.random.default_rng(3).uniform(0, 8, (50, 2))
        a, b = build_quadtree(points, (8, 8), capacity=2), build_quadtree(points, (8, 8), capacity=2)
        self.assertEqual(a.nodes, b.nodes)
        first = select_quadtree_matches(a, points, max_matches=8, residuals=np.arange(50)[::-1])
        second = select_quadtree_matches(b, points, max_matches=8, residuals=np.arange(50)[::-1])
        np.testing.assert_array_equal(first.original_indices, second.original_indices)

    def test_minimum_child_size(self):
        points = np.tile([0., 0.], (5, 1))
        tree = build_quadtree(points, (5, 7), capacity=1, min_cell_size=(2, 2))
        self.assertEqual(tree_statistics(tree)['maximum_reached_depth'], 1)
        self.assertTrue(build_quadtree(points, (1, 1), capacity=1).root.is_leaf)

    def test_invalid_parameters_and_scores(self):
        for options in ({'capacity': 0}, {'capacity': True}, {'max_depth': -1}, {'max_depth': 65},
                        {'max_depth': 1.5}, {'min_cell_size': (0, 1)}, {'min_cell_size': (np.nan, 1)}):
            with self.assertRaises(ValueError):
                build_quadtree(self.points, (8, 8), **options)
        for shape in ((0, 8), (1.5, 8), (True, 8)):
            with self.assertRaises(ValueError):
                build_quadtree(self.points, shape)
        tree = build_quadtree(self.points, (8, 8))
        for options in ({'confidence': [1]}, {'residuals': [1]}, {'confidence': [2]*4},
                        {'residuals': [-1]*4}, {'residuals': [np.nan]*4}, {'max_per_leaf': -1},
                        {'max_matches': True}):
            with self.assertRaises(ValueError):
                select_quadtree_matches(tree, self.points, **options)
        with self.assertRaises(ValueError):
            select_quadtree_matches(tree, self.points[:2])

    def test_zero_budgets_and_exhausted_leaves(self):
        points = np.array([[1, 1]]+[[5, 1]]*4+[[1, 5]]*4)
        tree = build_quadtree(points, (8, 8), capacity=4)
        for options in ({'max_per_leaf': 0}, {'max_matches': 0}):
            self.assertEqual(len(select_quadtree_matches(tree, points, **options).source), 0)
        selected = select_quadtree_matches(tree, points, max_per_leaf=3, max_matches=6)
        np.testing.assert_array_equal(np.unique(selected.leaf_ids, return_counts=True)[1], [1, 3, 2])

    def test_visualization_and_validation(self):
        tree = build_quadtree(self.points, (8, 8), capacity=1)
        fig, ax = plt.subplots()
        plot_quadtree(ax, np.arange(64).reshape(8, 8), tree, np.array([0, 1]))
        self.assertEqual(len(ax.patches), 4)
        fig.canvas.draw()
        with self.assertRaises(ValueError):
            plot_quadtree(ax, np.zeros((8, 8)), tree, np.array([0, 0]))

    def test_missing_comparison_and_output_protection(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(FileNotFoundError, 'M11 comparison report missing'):
                load_m11(Path(temp), {})
        root = Path(__file__).resolve().parents[1]
        with self.assertRaisesRegex(FileExistsError, 'overwrite'):
            run(root/'results/milestone_11_spatial_distribution')
        with self.assertRaisesRegex(ValueError, 'under project results'):
            run(root/'data/raw/forbidden')


if __name__ == '__main__':
    unittest.main()
