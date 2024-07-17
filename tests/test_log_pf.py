import unittest
import numpy as np
import jax.numpy as jnp
import os
import pandas as pd

from filter_forecast.particle_filter import log_pf
from filter_forecast.particle_filter.init_settings import InitSettings
from filter_forecast.particle_filter.output_handler import OutputHandler


class TestParticleCloud(unittest.TestCase):

    def setUp(self):
        self.settings = InitSettings(
            num_particles=10,
            population=1000,
            location_code="04",
            dt=1.0,
            seed_size=0.005,
        )
        self.particle_cloud = log_pf.ParticleCloud(self.settings)

    def test_initialization(self):
        self.assertEqual(
            self.particle_cloud.states.shape[0], self.settings.num_particles
        )
        self.assertEqual(
            self.particle_cloud.weights.shape[0], self.settings.num_particles
        )
        self.assertEqual(
            self.particle_cloud.betas.shape[0], self.settings.num_particles
        )

    def test_initial_state(self):
        initial_state = self.particle_cloud.get_initial_state()
        self.assertEqual(len(initial_state), 5)
        self.assertAlmostEqual(sum(initial_state), self.settings.population, delta=1e-5)

    def test_update_single_particle(self):
        initial_state = self.particle_cloud.states[0].copy()
        new_state = self.particle_cloud._update_single_particle(
            self.particle_cloud.states[0], 0, 0.1, self.settings.dt
        )
        self.particle_cloud.states = self.particle_cloud.states.at[0].set(new_state)
        updated_state = self.particle_cloud.states[0]
        self.assertFalse(jnp.array_equal(initial_state, updated_state))

    def test_update_all_particles(self):
        initial_states = self.particle_cloud.states.copy()
        self.particle_cloud.update_all_particles(0)
        updated_states = self.particle_cloud.states
        self.assertFalse(jnp.array_equal(initial_states, updated_states))

    def test_observation_class(self):
        hosp_cases = jnp.array([3, 5, 6, 3, 8, 9, 121, 7])
        observations = log_pf.ObservationData(observations=hosp_cases)
        self.assertEqual(observations.get_observation(2), 6)

    def test_compute_single_weight(self):
        reported_data = 17
        particle_estimates = [17, 23]
        weight1 = self.particle_cloud._compute_single_weight(
            reported_data, particle_estimates[0]
        )
        weight2 = self.particle_cloud._compute_single_weight(
            reported_data, particle_estimates[1]
        )
        # better estimate should have bigger weight
        self.assertTrue(
            weight1 > weight2, msg="The better estimate has a " "lower weight."
        )
        self.assertIsInstance(weight1, float)
        self.assertIsInstance(weight2, float)

    def test_compute_all_weights(self):
        self.particle_cloud.hosp_estimates = jnp.zeros(
            self.particle_cloud.settings.num_particles
        )
        self.particle_cloud.hosp_estimates = jnp.ones(
            self.particle_cloud.settings.num_particles
        )
        best_estimate_index = 1
        self.particle_cloud.hosp_estimates = self.particle_cloud.hosp_estimates.at[
            best_estimate_index
        ].set(10)
        reported_data = 20
        self.particle_cloud.compute_all_weights(reported_data)
        max_index = jnp.argmax(self.particle_cloud.weights)
        self.assertEqual(
            max_index,
            best_estimate_index,
            "The best estimate " "does not have the " "highest weight.",
        )

    def test_normalize_weights(self):
        self.particle_cloud.weights = jnp.ones(
            self.particle_cloud.settings.num_particles
        )
        self.particle_cloud.weights = self.particle_cloud.weights.at[1].set(5)
        high_weight_index = jnp.argmax(self.particle_cloud.weights)
        self.particle_cloud.normalize_weights()
        high_norm_weight_index = jnp.argmax(self.particle_cloud.weights)
        self.assertEqual(
            high_weight_index,
            high_norm_weight_index,
            "The index with the highest weight was not retained after normalization.",
        )


class TestOutputHandler(unittest.TestCase):

    def setUp(self):
        self.settings = InitSettings(num_particles=10, population=10000, location_code='04')
        self.runtime = 5
        self.handler = OutputHandler(self.settings, self.runtime)
        self.handler.set_destination_directory("/tmp")  # Set a temporary directory for testing

    def test_set_destination_directory(self):
        self.handler.set_destination_directory("/new_dir")
        self.assertEqual(self.handler.destination_dir, "/new_dir")

    def test_validate_betas_shape_correct(self):
        all_betas = np.random.rand(self.settings.num_particles, self.runtime)
        try:
            self.handler.validate_betas_shape(all_betas)
        except ValueError:
            self.fail("validate_betas_shape raised ValueError unexpectedly!")

    def test_validate_betas_shape_incorrect(self):
        all_betas = np.random.rand(self.settings.num_particles, self.runtime + 1)
        with self.assertRaises(ValueError):
            self.handler.validate_betas_shape(all_betas)

    def test_get_average_betas(self):
        all_betas = np.random.rand(self.settings.num_particles, self.runtime)
        self.handler.get_average_betas(all_betas)
        expected_avg_betas = np.mean(all_betas, axis=0)
        np.testing.assert_array_almost_equal(self.handler.avg_betas, expected_avg_betas)

    def test_output_average_betas(self):
        all_betas = np.random.rand(self.settings.num_particles, self.runtime)
        self.handler.get_average_betas(all_betas)
        self.handler.output_average_betas(all_betas)
        output_file = os.path.join(self.handler.destination_dir, "average_betas.csv")
        self.assertTrue(os.path.exists(output_file))
        df = pd.read_csv(output_file)
        np.testing.assert_array_almost_equal(df.values.flatten(), self.handler.avg_betas)

    def tearDown(self):
        # Clean up the temporary file created during the test
        output_file = os.path.join(self.handler.destination_dir, "average_betas.csv")
        if os.path.exists(output_file):
            os.remove(output_file)


if __name__ == "__main__":
    unittest.main()