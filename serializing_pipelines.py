'''
Serialization means converting a pipeline to a format that you can save on your disk and load later. 
It's especially useful because a serialized pipeline can be saved on disk or a database, get sent over a network and more.
Although it's possible to serialize into other formats too, 
Haystack supports YAML out of the box to make it easy for humans to make changes without the need to go back and forth with Python code. 
'''

from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.telemetry import tutorial_running

def enable_telemetry():
    tutorial_running(29)

def create_initial_pipeline():
    template = """
    Please create a summary about the following topic:
    {{ topic }}
    """
    builder = PromptBuilder(template=template)
    llm = HuggingFaceLocalGenerator(
        model="google/flan-t5-large",
        task="text2text-generation",
        generation_kwargs={"max_new_tokens": 150}
    )

    pipeline = Pipeline()
    pipeline.add_component(name="builder", instance=builder)
    pipeline.add_component(name="llm", instance=llm)
    pipeline.connect("builder", "llm")

    return pipeline

def run_initial_pipeline(pipeline, topic):
    result = pipeline.run(data={"builder": {"topic": topic}})
    return result["llm"]["replies"][0]

def serialize_pipeline(pipeline):
    return pipeline.dumps()

def edit_yaml_pipeline():
    return """
    components:
      builder:
        init_parameters:
          template: "\nPlease translate the following to French: \n{{ sentence }}\n"
        type: haystack.components.builders.prompt_builder.PromptBuilder
      llm:
        init_parameters:
          generation_kwargs:
            max_new_tokens: 150
          huggingface_pipeline_kwargs:
            device: cpu
            model: google/flan-t5-large
            task: text2text-generation
            token: null
          stop_words: null
        type: haystack.components.generators.hugging_face_local.HuggingFaceLocalGenerator
    connections:
    - receiver: llm.prompt
      sender: builder.prompt
    max_loops_allowed: 100
    metadata: {}
    """

def deserialize_pipeline(yaml_pipeline):
    return Pipeline.loads(yaml_pipeline)

def run_new_pipeline(pipeline, sentence):
    return pipeline.run(data={"builder": {"sentence": sentence}})

def main():
    enable_telemetry()

    # Create and run initial pipeline
    initial_pipeline = create_initial_pipeline()
    initial_result = run_initial_pipeline(initial_pipeline, "Climate change")
    print("Initial pipeline result:", initial_result)

    # Serialize pipeline to YAML
    yaml_pipeline = serialize_pipeline(initial_pipeline)
    print("\nSerialized pipeline:\n", yaml_pipeline)

    # Edit YAML pipeline
    edited_yaml_pipeline = edit_yaml_pipeline()
    print("\nEdited YAML pipeline:\n", edited_yaml_pipeline)

    # Deserialize edited YAML pipeline
    new_pipeline = deserialize_pipeline(edited_yaml_pipeline)

    # Run new pipeline
    new_result = run_new_pipeline(new_pipeline, "I love capybaras")
    print("\nNew pipeline result:", new_result)

if __name__ == "__main__":
    main()
